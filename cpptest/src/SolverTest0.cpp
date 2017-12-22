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

void testSolverParallelNormalize(Solver * solverDevPtr,
		CUDA_COMPLEX_TYPE * stateVectorDevPtr) {
testSolverParallelNormalizeKernel<<<1, 32>>>(solverDevPtr, stateVectorDevPtr);

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
		const CUDA_COMPLEX_TYPE * const matrixDevPtr, const int rows,
		const int columns, CUDA_COMPLEX_TYPE * stateVectorDevPtr,
		CUDA_COMPLEX_TYPE * resultVectorDevPtr) {
testSolverParallelMultMatrixVectorKernel<<<1, 32>>>(solverDevPtr, matrixDevPtr, rows, columns, stateVectorDevPtr, resultVectorDevPtr);

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
		const int csr3MatrixRowsNum,
		const CUDA_COMPLEX_TYPE * const csr3MatrixValuesDevPtr,
		const int * const csr3MatrixColumnsDevPtr,
		const int * const csr3MatrixRowIndexDevPtr,
		CUDA_COMPLEX_TYPE *vectorDevPtr, CUDA_COMPLEX_TYPE *resultDevPtr) {
testSolverParallelMultCSR3MatrixVectorKernel<<<1, 32>>>(solverDevPtr, csr3MatrixRowsNum,
		csr3MatrixValuesDevPtr, csr3MatrixColumnsDevPtr, csr3MatrixRowIndexDevPtr,
		vectorDevPtr, resultDevPtr);

							getLastCudaError("testSolverParallelMultCSR3MatrixVector");
}

__global__ void testSolverParallelCopyKernel(Solver * solverDevPtr,
		const CUDA_COMPLEX_TYPE * const source,
		CUDA_COMPLEX_TYPE * const dest) {
	solverDevPtr->parallelCopy(source, dest);
}

void testSolverParallelCopy(Solver * solverDevPtr,
		const CUDA_COMPLEX_TYPE * const source,
		CUDA_COMPLEX_TYPE * const dest) {
testSolverParallelCopyKernel<<<1, 32>>>(solverDevPtr, source, dest);

							getLastCudaError("testSolverParallelCopy");
}

__global__ void testSolverParallelCalcAlphaVectorKernel(Solver * solverDevPtr,
		const FLOAT_TYPE alpha, const CUDA_COMPLEX_TYPE * const vector,
		CUDA_COMPLEX_TYPE * const result) {
	solverDevPtr->parallelCalcAlphaVector(alpha, vector, result);
}

void testSolverParallelCalcAlphaVector(Solver * solverDevPtr,
		const FLOAT_TYPE alpha, const CUDA_COMPLEX_TYPE * const vector,
		CUDA_COMPLEX_TYPE * const result) {
testSolverParallelCalcAlphaVectorKernel<<<1, 32>>>(solverDevPtr, alpha, vector, result);

							getLastCudaError("testSolverParallelCalcAlphaVector");
}

__global__ void testSolverParallelCalcV1PlusAlphaV2Kernel(Solver * solverDevPtr,
		const CUDA_COMPLEX_TYPE * const v1, const FLOAT_TYPE alpha,
		const CUDA_COMPLEX_TYPE * const v2, CUDA_COMPLEX_TYPE * const result) {
	solverDevPtr->parallelCalcV1PlusAlphaV2(v1, alpha, v2, result);
}

void testSolverParallelCalcV1PlusAlphaV2(Solver * solverDevPtr,
		const CUDA_COMPLEX_TYPE * const v1, const FLOAT_TYPE alpha,
		const CUDA_COMPLEX_TYPE * const v2, CUDA_COMPLEX_TYPE * const result) {
testSolverParallelCalcV1PlusAlphaV2Kernel<<<1, 32>>>(solverDevPtr, v1, alpha, v2,result);

							getLastCudaError("testSolverParallelCalcV1PlusAlphaV2");
}

__global__ void testSolverSolverKernel(Solver * solverDevPtr) {
	solverDevPtr->solve();
}

void testSolverSolve(Solver * solverDevPtr) {
testSolverSolverKernel<<<1, 64>>>(solverDevPtr);

							getLastCudaError("testSolverSolve");
}

__global__ void testSolverParallelMakeJumpKernel(Solver * solverDevPtr) {
	solverDevPtr->parallelMakeJump();
}

void testSolverParallelMakeJump(Solver * solverDevPtr) {
	testSolverParallelMakeJumpKernel<<<1, 32>>>(solverDevPtr);

							getLastCudaError("testSolverSolve");
}
