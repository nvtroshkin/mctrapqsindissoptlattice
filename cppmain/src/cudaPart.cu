/*
 * cudaPart.cu
 *
 *  Created on: Dec 19, 2017
 *      Author: fakesci
 */

//Compute capability 1.x doesn't allow separate compilation

#include "precision-definition.h"

#include "Solver.cpp"

#ifdef TEST_MODE
#include "../../cpptest/src/SolverTest0.cpp"
#endif

__global__ void simulateKernel(int basisSize,
FLOAT_TYPE timeStep, int nTimeSteps, const CUDA_COMPLEX_TYPE * lDevPtr,
		int a1CSR3RowsNum, const CUDA_COMPLEX_TYPE * a1CSR3ValuesDevPtr,
		const int * a1CSR3ColumnsDevPtr, const int * a1CSR3RowIndexDevPtr,
		int a2CSR3RowsNum, const CUDA_COMPLEX_TYPE * a2CSR3ValuesDevPtr,
		const int * a2CSR3ColumnsDevPtr, const int * a2CSR3RowIndexDevPtr,
		int a3CSR3RowsNum, const CUDA_COMPLEX_TYPE * a3CSR3ValuesDevPtr,
		const int * a3CSR3ColumnsDevPtr, const int * a3CSR3RowIndexDevPtr,
		//block-local
		FLOAT_TYPE ** svNormThresholdDevPtr,
		FLOAT_TYPE ** sharedFloatDevPtr,
		CUDA_COMPLEX_TYPE *** sharedPointerDevPtr,
		CUDA_COMPLEX_TYPE ** k1DevPtr,
		CUDA_COMPLEX_TYPE ** k2DevPtr,
		CUDA_COMPLEX_TYPE ** k3DevPtr, CUDA_COMPLEX_TYPE ** k4DevPtr,
		CUDA_COMPLEX_TYPE ** prevStateDevPtr,
		CUDA_COMPLEX_TYPE ** curStateDevPtr) {

	Solver solver(basisSize, timeStep, nTimeSteps, lDevPtr, a1CSR3RowsNum,
			a1CSR3ValuesDevPtr, a1CSR3ColumnsDevPtr, a1CSR3RowIndexDevPtr, a2CSR3RowsNum,
			a2CSR3ValuesDevPtr, a2CSR3ColumnsDevPtr, a2CSR3RowIndexDevPtr, a3CSR3RowsNum,
			a3CSR3ValuesDevPtr, a3CSR3ColumnsDevPtr, a3CSR3RowIndexDevPtr,
			//block-local
			svNormThresholdDevPtr[blockIdx.x], sharedFloatDevPtr[blockIdx.x],
			sharedPointerDevPtr[blockIdx.x], k1DevPtr[blockIdx.x], k2DevPtr[blockIdx.x],
			k3DevPtr[blockIdx.x], k4DevPtr[blockIdx.x], prevStateDevPtr[blockIdx.x],
			curStateDevPtr[blockIdx.x]);
	solver.solve();
}

void simulate(const int nBlocks, const int nThreadsPerBlock, int basisSize,
FLOAT_TYPE timeStep, int nTimeSteps, const CUDA_COMPLEX_TYPE * lDevPtr,
		int a1CSR3RowsNum, const CUDA_COMPLEX_TYPE * a1CSR3ValuesDevPtr,
		const int * a1CSR3ColumnsDevPtr, const int * a1CSR3RowIndexDevPtr,
		int a2CSR3RowsNum, const CUDA_COMPLEX_TYPE * a2CSR3ValuesDevPtr,
		const int * a2CSR3ColumnsDevPtr, const int * a2CSR3RowIndexDevPtr,
		int a3CSR3RowsNum, const CUDA_COMPLEX_TYPE * a3CSR3ValuesDevPtr,
		const int * a3CSR3ColumnsDevPtr, const int * a3CSR3RowIndexDevPtr,
		//block-local
		FLOAT_TYPE ** svNormThresholdDevPtr,
		FLOAT_TYPE ** sharedFloatDevPtr,
		CUDA_COMPLEX_TYPE *** sharedPointerDevPtr,
		CUDA_COMPLEX_TYPE ** k1DevPtr,
		CUDA_COMPLEX_TYPE ** k2DevPtr,
		CUDA_COMPLEX_TYPE ** k3DevPtr, CUDA_COMPLEX_TYPE ** k4DevPtr,
		CUDA_COMPLEX_TYPE ** prevStateDevPtr,
		CUDA_COMPLEX_TYPE ** curStateDevPtr) {
	simulateKernel<<<nBlocks, nThreadsPerBlock>>>((int) basisSize, timeStep, (int) nTimeSteps, lDevPtr,
			a1CSR3RowsNum,a1CSR3ValuesDevPtr,
			a1CSR3ColumnsDevPtr, a1CSR3RowIndexDevPtr,
			a2CSR3RowsNum, a2CSR3ValuesDevPtr,
			a2CSR3ColumnsDevPtr, a2CSR3RowIndexDevPtr,
			a3CSR3RowsNum, a3CSR3ValuesDevPtr,
			a3CSR3ColumnsDevPtr, a3CSR3RowIndexDevPtr,
			//block-local
			svNormThresholdDevPtr, sharedFloatDevPtr, sharedPointerDevPtr, k1DevPtr,
			k2DevPtr, k3DevPtr, k4DevPtr, prevStateDevPtr, curStateDevPtr);
}
