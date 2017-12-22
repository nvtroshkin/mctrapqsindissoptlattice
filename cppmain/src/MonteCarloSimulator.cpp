/*
 * MonteCarloSimulator.cpp
 *
 *  Created on: Nov 29, 2017
 *      Author: fakesci
 */

#include <omp.h>
#include <iostream>
#include <cmath>
#include <exception>

#include "include/Model.h"
#include "include/MonteCarloSimulator.h"
#include "include/precision-definition.h"
#include "include/SimulationResult.h"
#include "Solver.h"
#include "utilities.h"

#include <cuda_runtime.h>
#include "helper_cuda.h"

//for the syntax checker
#ifndef __CUDA_ARCH__
extern const uint3 blockIdx;
#endif

template<typename T> T** initDev2DArray(T ** &dev2DArray, uint topLevelSize,
		uint bottomLevelSize) {
	//Allocate space on the host for each 1D device pointer
	T ** host1DDevPtrs = new T*[topLevelSize];

	//Allocate device space for the 1D arrays and write it to the host array of pointers
	for (int i = 0; i < topLevelSize; ++i) {
		checkCudaErrors(
				cudaMalloc((void** ) &(host1DDevPtrs[i]),
						bottomLevelSize * sizeof(T)));
	}

	//Allocate space on the device for the 2D array
	checkCudaErrors(
			cudaMalloc((void** ) &dev2DArray, topLevelSize * sizeof(T *)));

	//Copy the 1D device pointers from host to device
	checkCudaErrors(
			cudaMemcpy(dev2DArray, host1DDevPtrs, topLevelSize * sizeof(T *),
					cudaMemcpyHostToDevice));

	return host1DDevPtrs;
}

void initDevCSR3Matrix(CSR3Matrix * csr3Matrix,
CUDA_COMPLEX_TYPE ** valuesDevPtr, int ** columnsDevPtr,
		int ** rowIndexDevPtr) {
	int valuesNumber = csr3Matrix->rowIndex[csr3Matrix->rowsNumber];

	checkCudaErrors(
			cudaMalloc((void**) &valuesDevPtr, valuesNumber * sizeof(CUDA_COMPLEX_TYPE)));
	checkCudaErrors(
			cudaMemcpy(valuesDevPtr, csr3Matrix->values, valuesNumber * sizeof(CUDA_COMPLEX_TYPE), cudaMemcpyHostToDevice));

	checkCudaErrors(
			cudaMalloc((void** ) &columnsDevPtr, valuesNumber * sizeof(int)));
	checkCudaErrors(
			cudaMemcpy(columnsDevPtr, csr3Matrix->columns,
					valuesNumber * sizeof(int), cudaMemcpyHostToDevice));

	checkCudaErrors(
			cudaMalloc((void** ) &rowIndexDevPtr,
					(valuesNumber + 1) * sizeof(int)));
	checkCudaErrors(
			cudaMemcpy(rowIndexDevPtr, csr3Matrix->rowIndex,
					(csr3Matrix->rowsNumber + 1) * sizeof(int),
					cudaMemcpyHostToDevice));
}

__global__ void simulate0(int basisSize, FLOAT_TYPE timeStep, int nTimeSteps,
CUDA_COMPLEX_TYPE * l, int a1CSR3RowsNum,
CUDA_COMPLEX_TYPE * a1CSR3Values, int * a1CSR3Columns, int * a1CSR3RowIndex,
		int a2CSR3RowsNum,
		CUDA_COMPLEX_TYPE * a2CSR3Values, int * a2CSR3Columns,
		int * a2CSR3RowIndex, int a3CSR3RowsNum,
		CUDA_COMPLEX_TYPE * a3CSR3Values, int * a3CSR3Columns,
		int * a3CSR3RowIndex,
		//block-local
		FLOAT_TYPE ** svNormThresholdPtr,
		FLOAT_TYPE ** sharedFloatPtr, CUDA_COMPLEX_TYPE *** sharedPointerPtr,
		CUDA_COMPLEX_TYPE ** k1,
		CUDA_COMPLEX_TYPE ** k2,
		CUDA_COMPLEX_TYPE ** k3, CUDA_COMPLEX_TYPE ** k4,
		CUDA_COMPLEX_TYPE ** prevState, CUDA_COMPLEX_TYPE ** curState) {

	Solver solver(basisSize, timeStep, nTimeSteps, l, a1CSR3RowsNum,
			a1CSR3Values, a1CSR3Columns, a1CSR3RowIndex, a2CSR3RowsNum,
			a2CSR3Values, a2CSR3Columns, a2CSR3RowIndex, a3CSR3RowsNum,
			a3CSR3Values, a3CSR3Columns, a3CSR3RowIndex,
			//block-local
			svNormThresholdPtr[blockIdx.x], sharedFloatPtr[blockIdx.x],
			sharedPointerPtr[blockIdx.x], k1[blockIdx.x], k2[blockIdx.x],
			k3[blockIdx.x], k4[blockIdx.x], prevState[blockIdx.x],
			curState[blockIdx.x]/*, NULL, 0*/);
	solver.solve();
}

__host__ MonteCarloSimulator::MonteCarloSimulator(uint samplesNumber,
		Model &model) :
		basisSize(model.getBasisSize()), samplesNumber(samplesNumber), model(
				model), groundState(new CUDA_COMPLEX_TYPE[basisSize]()) {
	groundState[0].x = 1.0;
}

__host__ MonteCarloSimulator::~MonteCarloSimulator() {
	delete[] groundState;
}

__host__ SimulationResult *MonteCarloSimulator::simulate(FLOAT_TYPE timeStep,
		uint nTimeSteps, uint threadsPerBlock, uint nBlocks) {

	CUDA_COMPLEX_TYPE *lDevPtr;

	CUDA_COMPLEX_TYPE * a1CSR3ValuesDevPtr;
	int * a1CSR3ColumnsDevPtr;
	int * a1CSR3RowIndexDevPtr;

	CUDA_COMPLEX_TYPE * a2CSR3ValuesDevPtr;
	int * a2CSR3ColumnsDevPtr;
	int * a2CSR3RowIndexDevPtr;

	CUDA_COMPLEX_TYPE * a3CSR3ValuesDevPtr;
	int * a3CSR3ColumnsDevPtr;
	int * a3CSR3RowIndexDevPtr;

	//block-local
	FLOAT_TYPE ** svNormThresholdDevPtr;
	FLOAT_TYPE ** sharedFloatDevPtr;
	CUDA_COMPLEX_TYPE *** sharedPointerDevPtr;

	CUDA_COMPLEX_TYPE ** k1DevPtr;
	CUDA_COMPLEX_TYPE ** k2DevPtr;
	CUDA_COMPLEX_TYPE ** k3DevPtr;
	CUDA_COMPLEX_TYPE ** k4DevPtr;
	CUDA_COMPLEX_TYPE ** prevStateDevPtr;
	CUDA_COMPLEX_TYPE ** curStateDevPtr;	//also holds results

	uint basisSize = model.getBasisSize();

	//allocate device memory
	int lSize = basisSize * basisSize;
	checkCudaErrors(
			cudaMalloc((void**) &lDevPtr, lSize * sizeof(CUDA_COMPLEX_TYPE)));
	checkCudaErrors(
			cudaMemcpy(lDevPtr, model.getL(), lSize * sizeof(CUDA_COMPLEX_TYPE), cudaMemcpyHostToDevice));

	initDevCSR3Matrix(model.getA1InCSR3(), &a1CSR3ValuesDevPtr,
			&a1CSR3ColumnsDevPtr, &a1CSR3RowIndexDevPtr);
	initDevCSR3Matrix(model.getA2InCSR3(), &a2CSR3ValuesDevPtr,
			&a2CSR3ColumnsDevPtr, &a2CSR3RowIndexDevPtr);
	initDevCSR3Matrix(model.getA3InCSR3(), &a3CSR3ValuesDevPtr,
			&a3CSR3ColumnsDevPtr, &a3CSR3RowIndexDevPtr);

	//block-locals : one per block
	delete[] initDev2DArray(svNormThresholdDevPtr, nBlocks, 1);
	delete[] initDev2DArray(sharedFloatDevPtr, nBlocks, 1);
	delete[] initDev2DArray(sharedPointerDevPtr, nBlocks, 1);

	delete[] initDev2DArray(k1DevPtr, nBlocks, basisSize);
	delete[] initDev2DArray(k2DevPtr, nBlocks, basisSize);
	delete[] initDev2DArray(k3DevPtr, nBlocks, basisSize);
	delete[] initDev2DArray(k4DevPtr, nBlocks, basisSize);

	CUDA_COMPLEX_TYPE ** prevStateDevPtrsHostArray = initDev2DArray(
			prevStateDevPtr, nBlocks, basisSize);
	CUDA_COMPLEX_TYPE ** curStateDevPtrsHostArray = initDev2DArray(
			curStateDevPtr, nBlocks, basisSize);

//	CUDA_COMPLEX_TYPE *rowMajorL = model.getL();
//
//	//make it column major
//		CUDA_COMPLEX_TYPE columnMajorL[basisSize * basisSize];
//		for (int j = 0; j < basisSize; ++j) {
//			for (int i = 0; i < basisSize; ++i) {
//				columnMajorL[j * basisSize + i] = rowMajorL[i * basisSize + j];
//			}
//		}

	CUDA_COMPLEX_TYPE ** result = new CUDA_COMPLEX_TYPE *[samplesNumber];
	for (int i = 0; i < samplesNumber; ++i) {
		result[i] = new CUDA_COMPLEX_TYPE[basisSize];
	}

	uint nIterations = (samplesNumber - 1) / nBlocks + 1;

	uint actualNBlocks;
	uint resultIndex;
	for (int n = 0; n < nIterations; ++n) {
		//to not calculate unnecessary samples at the end
		actualNBlocks = std::min(nBlocks, samplesNumber - n * nBlocks);

		//prepare initial state
		for (int i = 0; i < actualNBlocks; ++i) {
			checkCudaErrors(
					cudaMemcpy(prevStateDevPtrsHostArray[i], groundState, basisSize * sizeof(CUDA_COMPLEX_TYPE), cudaMemcpyHostToDevice));
		}

simulate0<<<actualNBlocks, threadsPerBlock>>>((int) basisSize, timeStep, (int) nTimeSteps, lDevPtr,
			model.getA1InCSR3()->rowsNumber, a1CSR3ValuesDevPtr,
			a1CSR3ColumnsDevPtr, a1CSR3RowIndexDevPtr,
			model.getA2InCSR3()->rowsNumber, a2CSR3ValuesDevPtr,
			a2CSR3ColumnsDevPtr, a2CSR3RowIndexDevPtr,
			model.getA3InCSR3()->rowsNumber, a3CSR3ValuesDevPtr,
			a3CSR3ColumnsDevPtr, a3CSR3RowIndexDevPtr,
			//block-local
			svNormThresholdDevPtr, sharedFloatDevPtr, sharedPointerDevPtr, k1DevPtr,
			k2DevPtr, k3DevPtr, k4DevPtr, prevStateDevPtr, curStateDevPtr);

	//check for the kernel errors
				getLastCudaError("Monte-Carlo simulation failed");

		resultIndex = n * nBlocks;
		for (int i = 0; i < actualNBlocks; ++i) {
			checkCudaErrors(
					cudaMemcpy(result[resultIndex + i], curStateDevPtrsHostArray[i], basisSize * sizeof(CUDA_COMPLEX_TYPE), cudaMemcpyDeviceToHost));
		}
	}

	//freeing resources
	cudaDeviceReset();
	delete[] prevStateDevPtrsHostArray;
	delete[] curStateDevPtrsHostArray;

//#if defined(DEBUG_CONTINUOUS) || defined(DEBUG_JUMPS)
//	print(std::cout, "Result", result, basisSize, samplesNumber);
//#endif

	SimulationResult *simulationResult = new SimulationResult(result,
			samplesNumber, model);

	return simulationResult;
}

