/*
 *  Created on: Nov 24, 2017
 *      Author: fakesci
 */

#include "include/precision-definition.h"
#include "include/Solver.h"
#include "include/eval-params.h"

#include "cuda_runtime.h"
#include "curand_kernel.h"
#include "math.h"

#if defined(DEBUG_CONTINUOUS) || defined(DEBUG_JUMPS)
#include <utilities.h>
#define LOG_IF_APPROPRIATE(a) if(shouldPrintDebugInfo) (a)
#endif

//for the syntax checker
#ifndef __CUDA_ARCH__
extern const dim3 gridDim;
extern const uint3 blockIdx;
extern void __syncthreads();
extern float rsqrtf(float f);
#endif

#ifdef TEST_MODE
__host__
#endif
__device__ Solver::Solver(int basisSize, FLOAT_TYPE timeStep, int nTimeSteps,
CUDA_COMPLEX_TYPE * l, int a1CSR3RowsNum,
CUDA_COMPLEX_TYPE * a1CSR3Values, int * a1CSR3Columns, int * a1CSR3RowIndex,
		int a2CSR3RowsNum,
		CUDA_COMPLEX_TYPE * a2CSR3Values, int * a2CSR3Columns,
		int * a2CSR3RowIndex, int a3CSR3RowsNum,
		CUDA_COMPLEX_TYPE * a3CSR3Values, int * a3CSR3Columns,
		int * a3CSR3RowIndex,
		FLOAT_TYPE * svNormThresholdPtr,
		FLOAT_TYPE * sharedFloatPtr, CUDA_COMPLEX_TYPE ** sharedPointerPtr,
		CUDA_COMPLEX_TYPE *k1,
		CUDA_COMPLEX_TYPE *k2,
		CUDA_COMPLEX_TYPE *k3, CUDA_COMPLEX_TYPE *k4,
		CUDA_COMPLEX_TYPE *prevState, CUDA_COMPLEX_TYPE *curState/*, char * log,
		 uint logMaxSize*/) :
		tStep(timeStep), tHalfStep(0.5 * timeStep), tSixthStep(timeStep / 6.0), nTimeSteps(
				nTimeSteps), basisSize(basisSize),
#ifdef L_SPARSE
				lCSR3(
						model.getLInCSR3())
#else
				l(l)
#endif
						, a1CSR3RowsNum(a1CSR3RowsNum), a1CSR3Values(
						a1CSR3Values), a1CSR3Columns(a1CSR3Columns), a1CSR3RowIndex(
						a1CSR3RowIndex), a2CSR3RowsNum(a2CSR3RowsNum), a2CSR3Values(
						a2CSR3Values), a2CSR3Columns(a2CSR3Columns), a2CSR3RowIndex(
						a2CSR3RowIndex), a3CSR3RowsNum(a3CSR3RowsNum), a3CSR3Values(
						a3CSR3Values), a3CSR3Columns(a3CSR3Columns), a3CSR3RowIndex(
						a3CSR3RowIndex), svNormThresholdPtr(svNormThresholdPtr), sharedFloatPtr(
						sharedFloatPtr), sharedPointerPtr(sharedPointerPtr), k1(
						k1), k2(k2), k3(k3), k4(k4), prevState(prevState), curState(
						curState), log(log)/*, logMaxSize(logMaxSize), logSize(0)*/{
}

/**
 * Stores the final result in the curStep
 */
__device__ void Solver::solve() {

	CUDA_COMPLEX_TYPE * tempPointer;

	//Only the first thread does all the job - others are used in fork regions only
	//but they should go through the code to
	if (threadIdx.x == 0) {
		curand_init(
		ULONG_LONG_MAX / gridDim.x * blockIdx.x, 0ull, 0ull, &state);
		//get a random number for the calculation of the random waiting time
		//of the next jump
		*svNormThresholdPtr = getNextRandomFloat();
	}

	//Calculate each sample by the time axis
	for (int i = 0; i < nTimeSteps; ++i) {
#if defined(DEBUG_CONTINUOUS) || defined(DEBUG_JUMPS)
		if(threadIdx.x == 0) {
			shouldPrintDebugInfo = (i % TIME_STEPS_BETWEEN_DEBUG == 0);
			LOG_IF_APPROPRIATE("Step number" + *svNormThresholdPtr);
		}
#endif

		//Before the next jump there is deterministic evolution guided by
		//the Shroedinger's equation

		//A jump occurs between t(i) and t(i+1) when square of the state vector norm
		//at t(i+1)...
		//write the f function

		parallelRungeKuttaStep();

		//...falls below the threshold, which is a random number
		//uniformly distributed between [0,1] - svNormThreshold

		//if the state vector at t(i+1) has a less square of the norm then the threshold
		//try a self-written norm?

		if (threadIdx.x == 0) {
			*sharedFloatPtr = calcNormSquare(curState);
		}

#ifdef DEBUG_JUMPS
		LOG_IF_APPROPRIATE(
				consoleStream << "Norm: threshold = " << svNormThreshold
				<< ", current = " << norm2<< endl);
#endif

		__syncthreads();
		if (*svNormThresholdPtr > *sharedFloatPtr) {
#ifdef DEBUG_JUMPS
			consoleStream << "Jump" << endl;
			consoleStream << "Step: " << i << endl;
			consoleStream << "Norm^2: threshold = " << svNormThreshold
			<< ", current = " << norm2.real << endl;
#endif

			parallelMakeJump();

			if (threadIdx.x == 0) {
				//update the random time
				*svNormThresholdPtr = getNextRandomFloat();
			}
#ifdef DEBUG_JUMPS
			consoleStream << "New norm^2 threshold = " << svNormThreshold
			<< endl;
#endif
		}

		//update indices for all threads
		//(they have their own pointers but pointing to the same memory address)
		//the "restrict" pointers behaviour is maintained manually
		//through synchronization
		tempPointer = curState;
		curState = prevState;
		prevState = tempPointer;
	}

	//swap back - curState should hold the final result
	tempPointer = curState;
	curState = prevState;
	prevState = tempPointer;

#ifdef DEBUG_CONTINUOUS
//	LOG_IF_APPROPRIATE(print(consoleStream, "Psi(n+1)", curState, basisSize));
#endif

	//final state normalization
	parallelNormalizeVector(curState);

#ifdef DEBUG_CONTINUOUS
//	LOG_IF_APPROPRIATE(
//			print(consoleStream, "Normed Psi(n+1)", curState, basisSize));
#endif
}

__device__ inline void Solver::parallelRungeKuttaStep() {
	//uses nextState as a temporary storage vector

#ifdef DEBUG_CONTINUOUS
//	LOG_IF_APPROPRIATE(print(consoleStream, "prevState", prevState, basisSize));
#endif

	//See: https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
	//k1 = f(t_n, y_n);
	parallelMultLV(prevState, k1);

#ifdef DEBUG_CONTINUOUS
//	LOG_IF_APPROPRIATE(print(consoleStream, "k1", k1, basisSize));
#endif

	//k2 = f(t_n + h/2, y_n + h/2 * k1);
	//1: curState = y_n + h/2 * k1
	parallelCalcV1PlusAlphaV2(prevState, tHalfStep, k1, curState);
	//2: k2=L curState
	parallelMultLV(curState, k2);

#ifdef DEBUG_CONTINUOUS
//	LOG_IF_APPROPRIATE(print(consoleStream, "k2", k2, basisSize));
#endif

	//k3 = f(t_n + h/2, y_n + h/2 * k2)
	//1: curState = y_n + h/2 * k2
	parallelCalcV1PlusAlphaV2(prevState, tHalfStep, k2, curState);
	//2: k3=L curState
	parallelMultLV(curState, k3);

#ifdef DEBUG_CONTINUOUS
//	LOG_IF_APPROPRIATE(print(consoleStream, "k3", k3, basisSize));
#endif

	//k4 = f(t_n + h, y_n + h * k3);
	//1: curState = y_n + h * k3
	parallelCalcV1PlusAlphaV2(prevState, tStep, k3, curState);
	//2: k4=L curState
	parallelMultLV(curState, k4);

#ifdef DEBUG_CONTINUOUS
//	LOG_IF_APPROPRIATE(print(consoleStream, "k4", k4, basisSize));
#endif

	//y_(n+1) = y_n + (h/6)(k1 + 2k2 + 2k3 + k4)
	parallelCalcCurrentY();
}

__device__ inline void Solver::parallelCalcCurrentY() {
	//waiting for the main thread
	__syncthreads();

	//one block per cycle
	int blocksPerVector = (basisSize - 1) / blockDim.x + 1;
	int index;

#pragma unroll
	for (int i = 0; i < blocksPerVector; ++i) {
		index = threadIdx.x + i * blockDim.x;
		if (index < basisSize) {
			curState[index].x = prevState[index].x
					+ tSixthStep
							* (k1[index].x + 2 * k2[index].x + 2 * k3[index].x
									+ k4[index].x);
			curState[index].y = prevState[index].y
					+ tSixthStep
							* (k1[index].y + 2 * k2[index].y + 2 * k3[index].y
									+ k4[index].y);
		}
	}

	//ensure result is ready
	__syncthreads();
}

__device__ inline void Solver::parallelMakeJump() {
//then a jump is occurred between t(i) and t(i+1)
//let's suppose it was at time t(i)

#ifdef DEBUG_JUMPS
	print(consoleStream, "State at the jump moment", prevState, basisSize);
#endif

//calculate vectors and their norms after each type of jump
//the jump is made from the previous step state
//(in the first cavity, in the second or in the third)
	parallelMultCSR3MatrixVector(a1CSR3RowsNum, a1CSR3Values, a1CSR3Columns,
			a1CSR3RowIndex, prevState, k1);
	parallelMultCSR3MatrixVector(a2CSR3RowsNum, a2CSR3Values, a2CSR3Columns,
			a2CSR3RowIndex, prevState, k2);
	parallelMultCSR3MatrixVector(a3CSR3RowsNum, a3CSR3Values, a3CSR3Columns,
			a3CSR3RowIndex, prevState, k3);

	if (threadIdx.x == 0) {
		FLOAT_TYPE n12 = calcNormSquare(k1);
		FLOAT_TYPE n22 = calcNormSquare(k2);
		FLOAT_TYPE n32 = calcNormSquare(k3);

//calculate probabilities of each jump
		FLOAT_TYPE n2Sum = n12 + n22 + n32;
		FLOAT_TYPE p1 = n12 / n2Sum;
		FLOAT_TYPE p12 = (n12 + n22) / n2Sum;	//two first cavities together

#ifdef DEBUG_JUMPS
		print(consoleStream, "If jump will be in the first cavity", k1, basisSize);
		consoleStream << "it's norm^2: " << n12.real << endl;

		print(consoleStream, "If jump will be in the second cavity", k2, basisSize);
		consoleStream << "it's norm^2: " << n22.real << endl;

		print(consoleStream, "If jump will be in the third cavity", k3, basisSize);
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
			*sharedPointerPtr = k1;
			*sharedFloatPtr = n12;
		} else if (rnd < p12) {
#ifdef DEBUG_JUMPS
			consoleStream << "Jumped in the SECOND cavity" << endl;
#endif
			*sharedPointerPtr = k2;
			*sharedFloatPtr = n22;
		} else {
#ifdef DEBUG_JUMPS
			consoleStream << "Jumped in the THIRD cavity" << endl;
#endif
			//do not remove - it should be initialized (or cleared)
			*sharedPointerPtr = k3;
			*sharedFloatPtr = n32;
		}
	}

	parallelCopy((*sharedPointerPtr), curState);
	parallelNormalizeVector(curState);

#ifdef DEBUG_JUMPS
	print(consoleStream, "State vector after the jump and normalization",
			curState, basisSize);
#endif
}

__device__ inline void Solver::parallelMultLV(
		const CUDA_COMPLEX_TYPE * const vector,
		CUDA_COMPLEX_TYPE *result) {
#ifdef L_SPARSE
	//write down the sparse branch
	complex_mkl_cspblas_csrgemv("n", &basisSize, lCSR3->values, lCSR3->rowIndex,
			lCSR3->columns, vector, result);
#else
	//gather all threads
	__syncthreads();

	parallelMultMatrixVector(l, basisSize, basisSize, vector, result);

	//ensure that the result is ready
	__syncthreads();
#endif
}

__device__ inline void Solver::parallelMultMatrixVector(
		const CUDA_COMPLEX_TYPE * const matrix, const int rows,
		const int columns, const CUDA_COMPLEX_TYPE * const vector,
		CUDA_COMPLEX_TYPE * const result) {
	//waiting for the main thread
	__syncthreads();

	//one block per cycle
	uint blocksPerVector = (columns - 1) / blockDim.x + 1;
	uint index;
	uint rowStartIndex;

#pragma unroll
	for (uint i = 0; i < blocksPerVector; ++i) {
		index = threadIdx.x + i * blockDim.x;
		if (index < rows) {
			//clear the vector
			result[index].x = 0.0;
			result[index].y = 0.0;

			rowStartIndex = index * columns;
#pragma unroll
			for (uint j = 0; j < columns; ++j) {
				result[index].x += matrix[rowStartIndex + j].x * vector[j].x
						- matrix[rowStartIndex + j].y * vector[j].y;
				result[index].y += matrix[rowStartIndex + j].x * vector[j].y
						+ matrix[rowStartIndex + j].y * vector[j].x;
			}
		}
	}

	//ensure result is ready
	__syncthreads();
}

__device__ inline void Solver::parallelMultCSR3MatrixVector(
		const int csr3MatrixRowsNum,
		const CUDA_COMPLEX_TYPE * const csr3MatrixValues,
		const int * const csr3MatrixColumns,
		const int * const csr3MatrixRowIndex,
		const CUDA_COMPLEX_TYPE * const vector, CUDA_COMPLEX_TYPE * result) {
	//gather all threads
	__syncthreads();

	//one block per cycle
	int blocksPerVector = (basisSize - 1) / blockDim.x + 1;

#pragma unroll
	for (int i = 0; i < blocksPerVector; ++i) {
		uint row = threadIdx.x + i * blockDim.x;
		if (row < csr3MatrixRowsNum) {
			uint row_begin = csr3MatrixRowIndex[row];
			uint row_end = csr3MatrixRowIndex[row + 1];
			multiplyRow(row_end - row_begin, csr3MatrixColumns + row_begin,
					csr3MatrixValues + row_begin, vector, result[row]);
		}
	}

	//ensure that the result is ready
	__syncthreads();
}

__device__ inline void Solver::multiplyRow(uint rowsize,
		const int * const columnIndices,
		const CUDA_COMPLEX_TYPE * const matrixValues,
		const CUDA_COMPLEX_TYPE * const vector,
		CUDA_COMPLEX_TYPE &result) {
	result.x = 0.0;
	result.y = 0.0;
#pragma unroll
	for (uint column = 0; column < rowsize; ++column) {
		result.x += matrixValues[column].x * vector[columnIndices[column]].x
				- matrixValues[column].y * vector[columnIndices[column]].y;
		result.y += matrixValues[column].x * vector[columnIndices[column]].y
				+ matrixValues[column].y * vector[columnIndices[column]].x;
	}
}

__device__ inline FLOAT_TYPE Solver::calcNormSquare(
		const CUDA_COMPLEX_TYPE * const v) {
	FLOAT_TYPE temp = 0.0;
	for (int i = 0; i < basisSize; ++i) {
		//vary bad
		temp += v[i].x * v[i].x + v[i].y * v[i].y;
	}

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

#pragma unroll
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

#pragma unroll
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

#pragma unroll
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
CUDA_COMPLEX_TYPE *stateVector) {
//calculate norm
	if (threadIdx.x == 0) {
		*sharedFloatPtr = rsqrt(calcNormSquare(stateVector));
	}

	parallelCalcAlphaVector(*sharedFloatPtr, stateVector, stateVector);
}

#ifdef TEST_MODE
__device__ const FLOAT_TYPE _randomNumbers[7]= {
	0.99, // a jump
	0.5,// the second cavity wins
	0.98,// a jump
	0.1,// the first cavity wins
	0.99,// a jump
	0.9,// the third cavity wins
	0.0// next threshold (impossible to reach)
};

__device__ uint _randomNumberCounter = 0;
#endif

__device__ inline FLOAT_TYPE Solver::getNextRandomFloat() {
#ifdef TEST_MODE
	return _randomNumbers[_randomNumberCounter++];
#else
	return curand_uniform(&state);
#endif
}

//__device__ uint strlen(const char * str) {
//	uint len = 0;
//	while (str[len] != '\0') {
//		++len;
//	}
//}

//__device__ uint strcat(const char * str1, const char * str2) {
//
//}

//__device__ inline void Solver::printLog(const char * str) {
//	if (threadIdx.x == 0) {
//		uint strLen = strlen(str);
//		if (logMaxSize > logSize + strLen) {
//			//if there is enough space
//			for (int i = 0; i < strLen; ++i) {
//				log[logSize + i] = str[i];
//			}
//
//			logSize += strLen;
//
//			log[logSize] = '\0';
//		}
//	}
//}

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
