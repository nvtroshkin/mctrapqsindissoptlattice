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
		int a1CSR3RowsNum, const CUDA_COMPLEX_TYPE * a1CSR3Values,
		const int * a1CSR3Columns, const int * a1CSR3RowIndex,
		int a2CSR3RowsNum, const CUDA_COMPLEX_TYPE * a2CSR3Values,
		const int * a2CSR3Columns, const int * a2CSR3RowIndex,
		int a3CSR3RowsNum, const CUDA_COMPLEX_TYPE * a3CSR3Values,
		const int * a3CSR3Columns, const int * a3CSR3RowIndex,
		//non-const
		FLOAT_TYPE * sharedNormThresholdPtr,
		FLOAT_TYPE * sharedFloatPtr,
		CUDA_COMPLEX_TYPE ** sharedPointerPtr,
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
						, a1CSR3RowsNum(a1CSR3RowsNum), a1CSR3Values(
						a1CSR3Values), a1CSR3Columns(a1CSR3Columns), a1CSR3RowIndex(
						a1CSR3RowIndex), a2CSR3RowsNum(a2CSR3RowsNum), a2CSR3Values(
						a2CSR3Values), a2CSR3Columns(a2CSR3Columns), a2CSR3RowIndex(
						a2CSR3RowIndex), a3CSR3RowsNum(a3CSR3RowsNum), a3CSR3Values(
						a3CSR3Values), a3CSR3Columns(a3CSR3Columns), a3CSR3RowIndex(
						a3CSR3RowIndex),
				//shared
				sharedNormThresholdPtr(sharedNormThresholdPtr), sharedFloatPtr(
						sharedFloatPtr), sharedPointerPtr(sharedPointerPtr), sharedK1(
						sharedK1), sharedK2(sharedK2), sharedK3(sharedK3), sharedK4(
						sharedK4), sharedPrevState(sharedPrevState), sharedCurState(
						sharedCurState) {
}

/**
 * Stores the final result in the curStep
 */
__device__ void Solver::solve() {

	CUDA_COMPLEX_TYPE * const curStatePtr = sharedCurState;

	CUDA_COMPLEX_TYPE * tempPointer;

	//Only the first thread does all the job - others are used in fork regions only
	//but they should go through the code to

	__syncthreads();

	if (threadIdx.x == 0) {
		curand_init(ULONG_LONG_MAX / gridDim.x * blockIdx.x, 0ull, 0ull,
				&randomGeneratorState);
		//get a random number for the calculation of the random waiting time
		//of the next jump
		*sharedNormThresholdPtr = getNextRandomFloat();
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

		__syncthreads();

		if (threadIdx.x == 0) {
			*sharedFloatPtr = calcNormSquare(sharedCurState);
		}

		__syncthreads();

#ifdef DEBUG_JUMPS
		LOG_IF_APPROPRIATE(
				consoleStream << "Norm: threshold = " << svNormThreshold
				<< ", current = " << norm2<< endl);
#endif

		if (*sharedNormThresholdPtr > *sharedFloatPtr) {
#ifdef DEBUG_JUMPS
			consoleStream << "Jump" << endl;
			consoleStream << "Step: " << i << endl;
			consoleStream << "Norm^2: threshold = " << svNormThreshold
			<< ", current = " << norm2.real << endl;
#endif

			parallelMakeJump();

			__syncthreads();

			if (threadIdx.x == 0) {
				//update the random time
				*sharedNormThresholdPtr = getNextRandomFloat();
			}

			__syncthreads();

#ifdef DEBUG_JUMPS
			consoleStream << "New norm^2 threshold = " << svNormThreshold
			<< endl;
#endif
		}

		//update indices for all threads
		//(they have their own pointers but pointing to the same memory address)
		//the "restrict" pointers behaviour is maintained manually
		//through synchronization

		__syncthreads();

		if (threadIdx.x == 0) {
			tempPointer = sharedCurState;
			sharedCurState = sharedPrevState;
			sharedPrevState = tempPointer;
		}
		__syncthreads();
	}

	//swap back - sharedCurState should hold the final result

	__syncthreads();

	//Prepare the Solver for reusage
	if (threadIdx.x == 0) {
		if (curStatePtr != sharedCurState) {
			sharedPrevState = sharedCurState;
			sharedCurState = curStatePtr;
		}

		//else all is OK
	}

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

#ifdef DEBUG_JUMPS
	print(consoleStream, "State at the jump moment", sharedPrevState, basisSize);
#endif

//calculate vectors and their norms after each type of jump
//the jump is made from the previous step state
//(in the first cavity, in the second or in the third)
	parallelMultCSR3MatrixVector(a1CSR3Values, a1CSR3Columns,
			a1CSR3RowIndex, sharedPrevState, sharedK1);
	parallelMultCSR3MatrixVector(a2CSR3Values, a2CSR3Columns,
			a2CSR3RowIndex, sharedPrevState, sharedK2);
	parallelMultCSR3MatrixVector(a3CSR3Values, a3CSR3Columns,
			a3CSR3RowIndex, sharedPrevState, sharedK3);

	__syncthreads();

	if (threadIdx.x == 0) {
		FLOAT_TYPE n12 = calcNormSquare(sharedK1);
		FLOAT_TYPE n22 = calcNormSquare(sharedK2);
		FLOAT_TYPE n32 = calcNormSquare(sharedK3);

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
			*sharedPointerPtr = sharedK1;
			*sharedFloatPtr = n12;
		} else if (rnd < p12) {
#ifdef DEBUG_JUMPS
			consoleStream << "Jumped in the SECOND cavity" << endl;
#endif
			*sharedPointerPtr = sharedK2;
			*sharedFloatPtr = n22;
		} else {
#ifdef DEBUG_JUMPS
			consoleStream << "Jumped in the THIRD cavity" << endl;
#endif
			//do not remove - it should be initialized (or cleared)
			*sharedPointerPtr = sharedK3;
			*sharedFloatPtr = n32;
		}
	}

	__syncthreads();

	parallelCopy((*sharedPointerPtr), sharedCurState);
	parallelNormalizeVector(sharedCurState);

#ifdef DEBUG_JUMPS
	print(consoleStream, "State vector after the jump and normalization",
			sharedCurState, basisSize);
#endif
}

__device__ inline void Solver::parallelMultMatrixVector(
		const CUDA_COMPLEX_TYPE * const matrix, const int rows,
		const int columns, const CUDA_COMPLEX_TYPE * const vector,
		CUDA_COMPLEX_TYPE * const result) {
	//waiting for the main thread
	__syncthreads();

	multMatrixVector<BASIS_SIZE, CUDA_THREADS_PER_BLOCK, CUDA_MATRIX_VECTOR_ILP_COLUMN, CUDA_MATRIX_VECTOR_ILP_ROW>(matrix, vector, result);

	//ensure result is ready
	__syncthreads();
}

__device__ inline void Solver::parallelMultCSR3MatrixVector(
		const CUDA_COMPLEX_TYPE * const csr3MatrixValues,
		const int * const csr3MatrixColumns,
		const int * const csr3MatrixRowIndex,
		const CUDA_COMPLEX_TYPE * const vector, CUDA_COMPLEX_TYPE * result) {
	//gather all threads
	__syncthreads();

	multSparseMatrixVector<BASIS_SIZE, CUDA_THREADS_PER_BLOCK, CUDA_SPARSE_MATRIX_VECTOR_ILP_COLUMN>(csr3MatrixValues, csr3MatrixColumns,
			csr3MatrixRowIndex, vector, result);

	//ensure that the result is ready
	__syncthreads();
}

__device__ inline FLOAT_TYPE Solver::calcNormSquare(
		const CUDA_COMPLEX_TYPE * const v) {
	FLOAT_TYPE temp = 0.0;
#pragma unroll 2
	for (int i = 0; i < basisSize; ++i) {
		//vary bad
		temp += v[i].x * v[i].x + v[i].y * v[i].y;
	}

	//__syncthreads();
//	multVectorVector(v, v, &temp);
	//__syncthreads();
	return temp;
}

__device__ inline void Solver::parallelCalcAlphaVector(const FLOAT_TYPE alpha,
		const CUDA_COMPLEX_TYPE * const vector,
		CUDA_COMPLEX_TYPE * const result) {
	//waiting for the main thread
	__syncthreads();

	//one block per cycle
	int blocksPerVector = (basisSize - 1) / blockDim.x + 1;
	int index;

#pragma unroll 2
	for (int i = 0; i < blocksPerVector; ++i) {
		index = threadIdx.x + i * blockDim.x;
		if (index < basisSize) {
			result[index].x = alpha * vector[index].x;
			result[index].y = alpha * vector[index].y;
		}
	}

	//ensure result is ready
	__syncthreads();
}

__device__ inline void Solver::parallelCalcV1PlusAlphaV2(
		const CUDA_COMPLEX_TYPE * const v1, const FLOAT_TYPE alpha,
		const CUDA_COMPLEX_TYPE * const v2,
		CUDA_COMPLEX_TYPE * const result) {
	//waiting for the main thread
	__syncthreads();

	//one block per cycle
	int blocksPerVector = (basisSize - 1) / blockDim.x + 1;
	int index;

#pragma unroll 2
	for (int i = 0; i < blocksPerVector; ++i) {
		index = threadIdx.x + i * blockDim.x;
		if (index < basisSize) {
			result[index].x = v1[index].x + alpha * v2[index].x;
			result[index].y = v1[index].y + alpha * v2[index].y;
		}
	}

	//ensure result is ready
	__syncthreads();
}

__device__ inline void Solver::parallelCopy(
		const CUDA_COMPLEX_TYPE * const source,
		CUDA_COMPLEX_TYPE * const dest) {
	//waiting for the main thread
	__syncthreads();

	//one block per cycle
	int blocksPerVector = (basisSize - 1) / blockDim.x + 1;
	int index;

#pragma unroll 2
	for (int i = 0; i < blocksPerVector; ++i) {
		index = threadIdx.x + i * blockDim.x;
		if (index < basisSize) {
			dest[index] = source[index];
		}
	}

	//ensure result is ready
	__syncthreads();
}

__device__ inline void Solver::parallelNormalizeVector(
CUDA_COMPLEX_TYPE *sharedStateVector) {

	__syncthreads();

//calculate norm
	if (threadIdx.x == 0) {
		*sharedFloatPtr = rsqrt(calcNormSquare(sharedStateVector));
	}

	__syncthreads();

	parallelCalcAlphaVector(*sharedFloatPtr, sharedStateVector,
			sharedStateVector);
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

//template<typename T, const uint blk>
//__device__ void multMatrixVector(const T * __restrict__ matrix, const T * __restrict__ vector,
//		T * __restrict__ result, const uint nRows, const uint nx) {
//
//	const uint tid = threadIdx.x + blockIdx.x * blockDim.x;
//	const uint hor_blocks = (nx + blk - 1) / blk;
//
//	__shared__ T x_shared[blk];
//
//	register T y_val = 0.0;
//
//#pragma unroll
//	for (uint m = 0; m < hor_blocks; ++m) {
//
//		if ((m * blk + threadIdx.x) < nx) {
//			x_shared[threadIdx.x] = vector[threadIdx.x + m * blk];
//
//		} else {
//
//			x_shared[threadIdx.x] = 0.0f;
//		}
//
//		__syncthreads();
//
//#pragma unroll
//		for (uint e = 0; e < blk; ++e) {
//			y_val += matrix[tid + (e + blk * m) * nRows] * x_shared[e];
//		}
//
//		__syncthreads();
//	}
//
//	if (tid < nRows) {
//		result[tid] = y_val;
//	}
//
//}
//
//#pragma once
//template<typename T, const uint_t blk>
//__host__ void matvec_engine(const T * RESTRICT dA, const T * RESTRICT dx,
//		T * RESTRICT dy, const uint_t nRows, const uint_t nx) {
//
//	dim3 dim_grid((nRows + blk - 1) / blk);
//	dim3 dim_block(blk);
//
//	matvec_kernel<T, blk> <<<dim_grid, dim_block>>>(dA, dx, dy, nRows, nx);
//
//}
