/*
 * SolverTest.cu
 *
 *  Created on: Dec 19, 2017
 *      Author: fakesci
 */

#include <cuda_runtime.h>
#include "helper_cuda.h"

__global__ void testSolverParallelNormalizeKernel(Solver * solverDevPtr,
		CUDA_COMPLEX_TYPE * stateVectorDevPtr) {
	solverDevPtr->parallelNormalizeVector(stateVectorDevPtr);
}

void testSolverParallelNormalize(Solver * solverDevPtr, uint nThreadsPerBlock,
		CUDA_COMPLEX_TYPE * stateVectorDevPtr) {
testSolverParallelNormalizeKernel<<<1, nThreadsPerBlock>>>(solverDevPtr, stateVectorDevPtr);

			getLastCudaError("testSolverParallelNormalize");
}

__global__ void testSolverParallelMultMatrixVectorKernel(Solver * solverDevPtr,
		const CUDA_COMPLEX_TYPE * const matrixDevPtr, const int rows,
		const int columns, CUDA_COMPLEX_TYPE * stateVectorDevPtr,
		CUDA_COMPLEX_TYPE * resultVectorDevPtr) {
	solverDevPtr->parallelMultMatrixVector(matrixDevPtr, rows, columns,
			stateVectorDevPtr, resultVectorDevPtr);
}

void testSolverParallelMultMatrixVector(Solver * solverDevPtr,
		uint nThreadsPerBlock, const CUDA_COMPLEX_TYPE * const matrixDevPtr,
		const int rows, const int columns,
		CUDA_COMPLEX_TYPE * stateVectorDevPtr,
		CUDA_COMPLEX_TYPE * resultVectorDevPtr) {
testSolverParallelMultMatrixVectorKernel<<<1, nThreadsPerBlock>>>(solverDevPtr, matrixDevPtr, rows, columns, stateVectorDevPtr, resultVectorDevPtr);

			getLastCudaError("testSolverParallelMultMatrixVector");
}

__global__ void testSolverParallelMultCSR3MatrixVectorKernel(
		Solver * solverDevPtr, const int csr3MatrixRowsNum,
		const CUDA_COMPLEX_TYPE * const csr3MatrixValuesDevPtr,
		const int * const csr3MatrixColumnsDevPtr,
		const int * const csr3MatrixRowIndexDevPtr,
		CUDA_COMPLEX_TYPE *vectorDevPtr, CUDA_COMPLEX_TYPE *resultDevPtr) {
	solverDevPtr->parallelMultCSR3MatrixVector(csr3MatrixRowsNum,
			csr3MatrixValuesDevPtr, csr3MatrixColumnsDevPtr,
			csr3MatrixRowIndexDevPtr, vectorDevPtr, resultDevPtr);
}

void testSolverParallelMultCSR3MatrixVector(Solver * solverDevPtr,
		uint nThreadsPerBlock, const int csr3MatrixRowsNum,
		const CUDA_COMPLEX_TYPE * const csr3MatrixValuesDevPtr,
		const int * const csr3MatrixColumnsDevPtr,
		const int * const csr3MatrixRowIndexDevPtr,
		CUDA_COMPLEX_TYPE *vectorDevPtr, CUDA_COMPLEX_TYPE *resultDevPtr) {
testSolverParallelMultCSR3MatrixVectorKernel<<<1, nThreadsPerBlock>>>(solverDevPtr, csr3MatrixRowsNum,
		csr3MatrixValuesDevPtr, csr3MatrixColumnsDevPtr, csr3MatrixRowIndexDevPtr,
		vectorDevPtr, resultDevPtr);

			getLastCudaError("testSolverParallelMultCSR3MatrixVector");
}

__global__ void testSolverParallelCopyKernel(Solver * solverDevPtr,
		const CUDA_COMPLEX_TYPE * const source,
		CUDA_COMPLEX_TYPE * const dest) {
	solverDevPtr->parallelCopy(source, dest);
}

void testSolverParallelCopy(Solver * solverDevPtr, uint nThreadsPerBlock,
		const CUDA_COMPLEX_TYPE * const source,
		CUDA_COMPLEX_TYPE * const dest) {
testSolverParallelCopyKernel<<<1, nThreadsPerBlock>>>(solverDevPtr, source, dest);

			getLastCudaError("testSolverParallelCopy");
}

__global__ void testSolverParallelCalcAlphaVectorKernel(Solver * solverDevPtr,
		const FLOAT_TYPE alpha, const CUDA_COMPLEX_TYPE * const vector,
		CUDA_COMPLEX_TYPE * const result) {
	solverDevPtr->parallelCalcAlphaVector(alpha, vector, result);
}

void testSolverParallelCalcAlphaVector(Solver * solverDevPtr,
		uint nThreadsPerBlock, const FLOAT_TYPE alpha,
		const CUDA_COMPLEX_TYPE * const vector,
		CUDA_COMPLEX_TYPE * const result) {
testSolverParallelCalcAlphaVectorKernel<<<1, nThreadsPerBlock>>>(solverDevPtr, alpha, vector, result);

			getLastCudaError("testSolverParallelCalcAlphaVector");
}

__global__ void testSolverParallelCalcV1PlusAlphaV2Kernel(Solver * solverDevPtr,
		const CUDA_COMPLEX_TYPE * const v1, const FLOAT_TYPE alpha,
		const CUDA_COMPLEX_TYPE * const v2, CUDA_COMPLEX_TYPE * const result) {
	solverDevPtr->parallelCalcV1PlusAlphaV2(v1, alpha, v2, result);
}

void testSolverParallelCalcV1PlusAlphaV2(Solver * solverDevPtr,
		uint nThreadsPerBlock, const CUDA_COMPLEX_TYPE * const v1,
		const FLOAT_TYPE alpha, const CUDA_COMPLEX_TYPE * const v2,
		CUDA_COMPLEX_TYPE * const result) {
testSolverParallelCalcV1PlusAlphaV2Kernel<<<1, nThreadsPerBlock>>>(solverDevPtr, v1, alpha, v2,result);

			getLastCudaError("testSolverParallelCalcV1PlusAlphaV2");
}

__device__ void initRandomNumbersForTest(bool noJumps) {
	if (threadIdx.x == 0) {
		if (noJumps) {
			_randomNumberCounter = 0;

			for (int i = 0; i < 7; ++i) {
				_randomNumbers[i] = 0.0;
			}
		} else {
			_randomNumberCounter = 0;

			_randomNumbers[0] = 0.99; // a jump
			_randomNumbers[1] = 0.5; // the second cavity wins
			_randomNumbers[2] = 0.98; // a jump
			_randomNumbers[3] = 0.1; // the first cavity wins
			_randomNumbers[4] = 0.99; // a jump
			_randomNumbers[5] = 0.9; // the third cavity wins
			_randomNumbers[6] = 0.0; // next threshold (impossible to reach)
		}
	}
}

__device__ void initRandomNumbersForTest(FLOAT_TYPE randomNumber) {
	if (threadIdx.x == 0) {
		_randomNumberCounter = 0;
		_randomNumbers[0] = randomNumber;
	}
}

__global__ void testSolverSolverKernel(Solver * solverDevPtr, bool noJumps) {
	initRandomNumbersForTest(noJumps);

	solverDevPtr->solve();
}

void testSolverSolve(Solver * solverDevPtr, uint nThreadsPerBlock,
		bool noJumps) {
testSolverSolverKernel<<<1, nThreadsPerBlock>>>(solverDevPtr, noJumps);

			getLastCudaError("testSolverSolve");
}

__global__ void testSolverParallelMakeJumpKernel(Solver * solverDevPtr,
		FLOAT_TYPE randomNumber) {
	initRandomNumbersForTest(randomNumber);

	solverDevPtr->parallelMakeJump();
}

void testSolverParallelMakeJump(Solver * solverDevPtr, uint nThreadsPerBlock,
		FLOAT_TYPE randomNumber) {
testSolverParallelMakeJumpKernel<<<1, nThreadsPerBlock>>>(solverDevPtr,randomNumber);

			getLastCudaError("testSolverSolve");
}
