/*
 *  Created on: Nov 24, 2017
 *      Author: fakesci
 */

#include "curand_kernel.h"

#include "include/precision-definition.h"
#include "include/Solver.h"
#include "include/eval-params.h"
#include "system-constants.h"
#include "custommath.h"

#if defined(DEBUG_CONTINUOUS) || defined(DEBUG_JUMPS)
#include <utilities.h>
#define LOG_IF_APPROPRIATE(a) if(shouldPrintDebugInfo) (a)
#endif

__host__ __device__ Solver::Solver(int basisSize, FLOAT_TYPE timeStep,
		int nTimeSteps, const CUDA_COMPLEX_TYPE * rungeKuttaOperator,
		const CUDA_COMPLEX_TYPE * a1CSR3Values, const int * a1CSR3Columns,
		const int * a1CSR3RowIndex, const CUDA_COMPLEX_TYPE * a2CSR3Values,
		const int * a2CSR3Columns, const int * a2CSR3RowIndex,
		const CUDA_COMPLEX_TYPE * a3CSR3Values, const int * a3CSR3Columns,
		const int * a3CSR3RowIndex,
		//non-const
		CUDA_COMPLEX_TYPE *sharedK1, CUDA_COMPLEX_TYPE *sharedK2,
		CUDA_COMPLEX_TYPE *sharedK3, CUDA_COMPLEX_TYPE *sharedK4,
		CUDA_COMPLEX_TYPE *sharedPrevState,
		CUDA_COMPLEX_TYPE *sharedCurState) :
		tStep(timeStep), nTimeSteps(nTimeSteps), basisSize(basisSize),
#ifdef L_SPARSE
				lCSR3(
						model.getLInCSR3())
#else
				rungeKuttaOperator(rungeKuttaOperator)
#endif
						, a1CSR3Values(a1CSR3Values), a1CSR3Columns(
						a1CSR3Columns), a1CSR3RowIndex(a1CSR3RowIndex), a2CSR3Values(
						a2CSR3Values), a2CSR3Columns(a2CSR3Columns), a2CSR3RowIndex(
						a2CSR3RowIndex), a3CSR3Values(a3CSR3Values), a3CSR3Columns(
						a3CSR3Columns), a3CSR3RowIndex(a3CSR3RowIndex),
				//shared
				sharedK1(sharedK1), sharedK2(sharedK2), sharedK3(sharedK3), sharedK4(
						sharedK4), sharedPrevState(sharedPrevState), sharedCurState(
						sharedCurState) {
}

/**
 * Stores the final result in the curStep
 */
__device__ void Solver::solve() {

	CUDA_COMPLEX_TYPE * const __restrict__ curStatePtr = sharedCurState;
	CUDA_COMPLEX_TYPE * __restrict__ tempPointer;

	static __shared__ FLOAT_TYPE sharedNorm2Threshold;
	static __shared__ FLOAT_TYPE sharedCurrentNorm2;

	//Only the first thread does all the job - others are used in fork regions only
	//but they should go through the code to

	if (threadIdx.x == 0) {
		curand_init(ULONG_LONG_MAX / gridDim.x * blockIdx.x, 0ull, 0ull,
				&randomGeneratorState);
		//get a random number for the calculation of the random waiting time
		//of the next jump
		sharedNorm2Threshold = getNextRandomFloat();
	}

	__syncthreads();

	//Calculate each sample by the time axis
	for (int i = 0; i < nTimeSteps; ++i) {
#if defined(DEBUG_CONTINUOUS) || defined(DEBUG_JUMPS)
		shouldPrintDebugInfo = (i % TIME_STEPS_BETWEEN_DEBUG == 0);
		LOG_IF_APPROPRIATE("Step number " + std::to_string(nTimeSteps));
#endif

		//Before the next jump there is deterministic evolution guided by
		//the Shroedinger's equation

		//A jump occurs between t(i) and t(i+1) when square of the state vector norm
		//at t(i+1)...
		//write the f function

		//y_(n+1) = 1 + L (h + L (0.5 h^2 + L (1/6 h^3 + 1/24 h^4 L))) yn
		parallelMultMatrixVector(rungeKuttaOperator, basisSize, basisSize,
				sharedPrevState, sharedCurState);

		//...falls below the threshold, which is a random number
		//uniformly distributed between [0,1] - svNormThreshold

		//if the state vector at t(i+1) has a less square of the norm then the threshold
		//try a self-written norm?

		FLOAT_TYPE norm2 = parallelCalcNormSquare(sharedCurState);
		if (threadIdx.x == 0) {
			//because it is shared
			sharedCurrentNorm2 = norm2;
		}

		__syncthreads();

#ifdef DEBUG_JUMPS
		LOG_IF_APPROPRIATE(
				consoleStream << "Norm: threshold = " << svNormThreshold
				<< ", current = " << norm2<< endl);
#endif

		if (sharedNorm2Threshold > sharedCurrentNorm2) {
#ifdef DEBUG_JUMPS
			consoleStream << "Jump" << endl;
			consoleStream << "Step: " << i << endl;
			consoleStream << "Norm^2: threshold = " << svNormThreshold
			<< ", current = " << norm2.real << endl;
#endif

			parallelMakeJump();

			if (threadIdx.x == 0) {
				sharedNorm2Threshold = getNextRandomFloat();
			}

			__syncthreads();

#ifdef DEBUG_JUMPS
			consoleStream << "New norm^2 threshold = " << svNormThreshold
			<< endl;
#endif
		}

		//update indices for all threads
		//(they have their own pointers but pointing to the same memory address)
		//the "restrict" pointers behavior is maintained manually
		//through synchronization

		if (threadIdx.x == 0) {
			tempPointer = sharedCurState;
			sharedCurState = sharedPrevState;
			sharedPrevState = tempPointer;
		}

		//because of swapping the variables
		__syncthreads();
	}

	//Prepare the Solver for reusage
	//sharedCurState should hold the final result
	if (threadIdx.x == 0) {
		if (curStatePtr != sharedCurState) {
			sharedPrevState = sharedCurState;
			sharedCurState = curStatePtr;
		}

		//else all is OK
	}

	//because of the possible swapping of the variables
	__syncthreads();

#ifdef DEBUG_CONTINUOUS
	LOG_IF_APPROPRIATE(print(consoleStream, "Psi(n+1)", sharedCurState, basisSize));
#endif

	//final state normalization (because we are swapping the pointers inside - should use a local one
	//which is equal to the host's)
	parallelNormalizeVector(sharedCurState);

#ifdef DEBUG_CONTINUOUS
	LOG_IF_APPROPRIATE(
			print(consoleStream, "Normed Psi(n+1)", sharedCurState, basisSize));
#endif
}

__device__ inline void Solver::parallelMakeJump() {
//then a jump is occurred between t(i) and t(i+1)
//let's suppose it was at time t(i)

	__syncthreads();

#ifdef DEBUG_JUMPS
	print(consoleStream, "State at the jump moment", sharedPrevState, basisSize);
#endif

//calculate vectors and their norms after each type of jump
//the jump is made from the previous step state
//(in the first cavity, in the second or in the third)
	parallelMultCSR3MatrixVector(a1CSR3Values, a1CSR3Columns, a1CSR3RowIndex,
			sharedPrevState, sharedK1);
	parallelMultCSR3MatrixVector(a2CSR3Values, a2CSR3Columns, a2CSR3RowIndex,
			sharedPrevState, sharedK2);
	parallelMultCSR3MatrixVector(a3CSR3Values, a3CSR3Columns, a3CSR3RowIndex,
			sharedPrevState, sharedK3);

	//only 0 thread will have the value
	FLOAT_TYPE n12 = parallelCalcNormSquare(sharedK1);
	FLOAT_TYPE n22 = parallelCalcNormSquare(sharedK2);
	FLOAT_TYPE n32 = parallelCalcNormSquare(sharedK3);

	static __shared__ CUDA_COMPLEX_TYPE * sharedStateAfterJump;

	if (threadIdx.x == 0) {
//calculate probabilities of each jump
		FLOAT_TYPE n2Sum = n12 + n22 + n32;
		FLOAT_TYPE p1 = n12 / n2Sum;
		FLOAT_TYPE p12 = (n12 + n22) / n2Sum;	//two first cavities together

#ifdef DEBUG_JUMPS
		print(consoleStream, "If jump will be in the first cavity", sharedK1, basisSize);
		consoleStream << "it's norm^2: " << n12.real << endl;

		print(consoleStream, "If jump will be in the second cavity", sharedK2, basisSize);
		consoleStream << "it's norm^2: " << n22.real << endl;

		print(consoleStream, "If jump will be in the third cavity", sharedK3, basisSize);
		consoleStream << "it's norm^2: " << n32.real << endl;

		consoleStream << "Probabilities of the jumps: in the first = " << p1
		<< ", (the first + the second) = " << p12 << ", third = " << 1-p12 << endl;
#endif

//choose which jump is occurred,
		FLOAT_TYPE rnd = getNextRandomFloat();

		if (rnd < p1) {
#ifdef DEBUG_JUMPS
			consoleStream << "It jumped in the FIRST cavity" << endl;
#endif
			sharedStateAfterJump = sharedK1;
		} else if (rnd < p12) {
#ifdef DEBUG_JUMPS
			consoleStream << "Jumped in the SECOND cavity" << endl;
#endif
			sharedStateAfterJump = sharedK2;
		} else {
#ifdef DEBUG_JUMPS
			consoleStream << "Jumped in the THIRD cavity" << endl;
#endif
			//do not remove - it should be initialized (or cleared)
			sharedStateAfterJump = sharedK3;
		}
	}

	__syncthreads();

	parallelCopy(sharedStateAfterJump, sharedCurState);
	parallelNormalizeVector(sharedCurState);

#ifdef DEBUG_JUMPS
	print(consoleStream, "State vector after the jump and normalization",
			sharedCurState, basisSize);
#endif

	__syncthreads();
}

__device__ inline void Solver::parallelMultMatrixVector(
		const CUDA_COMPLEX_TYPE * const __restrict__ matrix, const int rows,
		const int columns, const CUDA_COMPLEX_TYPE * const __restrict__ vector,
		CUDA_COMPLEX_TYPE * const __restrict__ result) {

	__syncthreads();

	multMatrixVector<BASIS_SIZE, CUDA_THREADS_PER_BLOCK,
			CUDA_MATRIX_VECTOR_ILP_COLUMN, CUDA_MATRIX_VECTOR_ILP_ROW, false>(
			matrix, vector, result);

	__syncthreads();
}

__device__ inline void Solver::parallelMultCSR3MatrixVector(
		const CUDA_COMPLEX_TYPE * const __restrict__ csr3MatrixValues,
		const int * const __restrict__ csr3MatrixColumns,
		const int * const __restrict__ csr3MatrixRowIndex,
		const CUDA_COMPLEX_TYPE * const __restrict__ vector,
		CUDA_COMPLEX_TYPE * __restrict__ result) {

	__syncthreads();

	multSparseMatrixVector<BASIS_SIZE, CUDA_THREADS_PER_BLOCK,
			CUDA_SPARSE_MATRIX_VECTOR_ILP_COLUMN>(csr3MatrixValues,
			csr3MatrixColumns, csr3MatrixRowIndex, vector, result);

	__syncthreads();
}

__device__ inline FLOAT_TYPE Solver::parallelCalcNormSquare(
		const CUDA_COMPLEX_TYPE * const __restrict__ v) {
	__syncthreads();

	CUDA_COMPLEX_TYPE temp;
	multVectorVector<BASIS_SIZE, CUDA_THREADS_PER_BLOCK,
			CUDA_MATRIX_VECTOR_ILP_COLUMN, true>(v, v, &temp);

	__syncthreads();

	return temp.x;
}

__device__ inline void Solver::parallelCalcAlphaVector(const FLOAT_TYPE alpha,
		const CUDA_COMPLEX_TYPE * const __restrict__ vector,
		CUDA_COMPLEX_TYPE * const __restrict__ result) {
	//waiting for the main thread
	__syncthreads();

	//one block per cycle
#pragma unroll 2
	for (int i = 0, index = threadIdx.x; i < BLOCKS_PER_VECTOR; ++i, index +=
			CUDA_THREADS_PER_BLOCK) {
		if (index < BASIS_SIZE) {
			result[index].x = alpha * vector[index].x;
			result[index].y = alpha * vector[index].y;
		}
	}

	//ensure result is ready
	__syncthreads();
}

__device__ inline void Solver::parallelCalcV1PlusAlphaV2(
		const CUDA_COMPLEX_TYPE * const __restrict__ v1, const FLOAT_TYPE alpha,
		const CUDA_COMPLEX_TYPE * const __restrict__ v2,
		CUDA_COMPLEX_TYPE * const __restrict__ result) {
	//waiting for the main thread
	__syncthreads();

#pragma unroll 2
	for (int i = 0, index = threadIdx.x; i < BLOCKS_PER_VECTOR; ++i, index +=
			CUDA_THREADS_PER_BLOCK) {
		if (index < BASIS_SIZE) {
			result[index].x = v1[index].x + alpha * v2[index].x;
			result[index].y = v1[index].y + alpha * v2[index].y;
		}
	}

	//ensure result is ready
	__syncthreads();
}

__device__ inline void Solver::parallelCopy(
		const CUDA_COMPLEX_TYPE * const __restrict__ source,
		CUDA_COMPLEX_TYPE * const __restrict__ dest) {

	__syncthreads();

#pragma unroll 2
	for (int i = 0, index = threadIdx.x; i < BLOCKS_PER_VECTOR; ++i, index +=
			CUDA_THREADS_PER_BLOCK) {
		if (index < BASIS_SIZE) {
			dest[index] = source[index];
		}
	}

	__syncthreads();
}

__device__ inline void Solver::parallelNormalizeVector(
CUDA_COMPLEX_TYPE * __restrict__ sharedStateVector) {

	__syncthreads();

	static __shared__ FLOAT_TYPE sharedFloat;

//calculate norm
	FLOAT_TYPE norm2 = parallelCalcNormSquare(sharedStateVector);
	if (threadIdx.x == 0) {
		sharedFloat = rsqrt(norm2);
	}

	//because of the argument
	__syncthreads();

	parallelCalcAlphaVector(sharedFloat, sharedStateVector, sharedStateVector);
}

#ifdef TEST_MODE
__device__ FLOAT_TYPE _randomNumbers[1000] = {0.0};	// no jumps

__device__ uint _randomNumberCounter = 0;
#endif

__device__ inline FLOAT_TYPE Solver::getNextRandomFloat() {
#ifdef TEST_MODE
	return _randomNumbers[_randomNumberCounter++];
#else
	return curand_uniform(&randomGeneratorState);
#endif
}
