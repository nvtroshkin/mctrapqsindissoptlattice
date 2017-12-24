/*
 * SolverFactory.cu
 *
 *  Created on: Dec 20, 2017
 *      Author: fakesci
 */
#include <vector>
#include <stdexcept>

#include "cuda_runtime.h"
#include "helper_cuda.h"

#include "Solver.h"

#include "SolverContext.h"

inline void _initDevCSR3Matrix(CSR3Matrix * csr3Matrix,
CUDA_COMPLEX_TYPE ** valuesDevPtr, int ** columnsDevPtr,
		int ** rowIndexDevPtr) {
	const uint rowsNumber = csr3Matrix->rowsNumber;
	const int valuesNumber = csr3Matrix->rowIndex[rowsNumber];

	checkCudaErrors(
			cudaMalloc((void**) valuesDevPtr, valuesNumber * sizeof(CUDA_COMPLEX_TYPE)));
	checkCudaErrors(
			cudaMemcpy(*valuesDevPtr, csr3Matrix->values, valuesNumber * sizeof(CUDA_COMPLEX_TYPE), cudaMemcpyHostToDevice));

	checkCudaErrors(
			cudaMalloc((void** ) columnsDevPtr, valuesNumber * sizeof(int)));
	checkCudaErrors(
			cudaMemcpy(*columnsDevPtr, csr3Matrix->columns,
					valuesNumber * sizeof(int), cudaMemcpyHostToDevice));

	checkCudaErrors(
			cudaMalloc((void** ) rowIndexDevPtr,
					(rowsNumber + 1) * sizeof(int)));
	checkCudaErrors(
			cudaMemcpy(*rowIndexDevPtr, csr3Matrix->rowIndex,
					(rowsNumber + 1) * sizeof(int), cudaMemcpyHostToDevice));
}

SolverContext::SolverContext(uint maxSolvers, FLOAT_TYPE timeStep,
		uint nTimeSteps, Model &model) :
		maxSolvers(maxSolvers), basisSize(model.getBasisSize()), timeStep(
				timeStep), nTimeSteps(nTimeSteps), a1CSR3RowsNum(
				model.getA1InCSR3()->rowsNumber), a2CSR3RowsNum(
				model.getA2InCSR3()->rowsNumber), a3CSR3RowsNum(
				model.getA3InCSR3()->rowsNumber) {
	svNormThresholdDevPtrs = new std::vector<FLOAT_TYPE *>();
	svNormThresholdDevPtrs->reserve(maxSolvers);

	sharedFloatDevPtrs = new std::vector<FLOAT_TYPE *>();
	sharedFloatDevPtrs->reserve(maxSolvers);

	sharedPointerDevPtrs = new std::vector<CUDA_COMPLEX_TYPE **>();
	sharedPointerDevPtrs->reserve(maxSolvers);

	k1DevPtrs = new std::vector<CUDA_COMPLEX_TYPE *>();
	k1DevPtrs->reserve(maxSolvers);

	k2DevPtrs = new std::vector<CUDA_COMPLEX_TYPE *>();
	k2DevPtrs->reserve(maxSolvers);

	k3DevPtrs = new std::vector<CUDA_COMPLEX_TYPE *>();
	k3DevPtrs->reserve(maxSolvers);

	k4DevPtrs = new std::vector<CUDA_COMPLEX_TYPE *>();
	k4DevPtrs->reserve(maxSolvers);

	prevStateDevPtrs = new std::vector<CUDA_COMPLEX_TYPE *>();
	prevStateDevPtrs->reserve(maxSolvers);

	curStateDevPtrs = new std::vector<CUDA_COMPLEX_TYPE *>();
	curStateDevPtrs->reserve(maxSolvers);

	solverPtrs = new std::vector<Solver *>();
	solverPtrs->reserve(maxSolvers);

	solverDevPtrs = new std::vector<Solver *>();
	solverDevPtrs->reserve(maxSolvers);

	//Global state
	int lSize = basisSize * basisSize;
	checkCudaErrors(
			cudaMalloc((void**) &lDevPtr, lSize * sizeof(CUDA_COMPLEX_TYPE)));
	checkCudaErrors(
			cudaMemcpy(lDevPtr, model.getL(), lSize * sizeof(CUDA_COMPLEX_TYPE), cudaMemcpyHostToDevice));

	_initDevCSR3Matrix(model.getA1InCSR3(), &a1CSR3ValuesDevPtr,
			&a1CSR3ColumnsDevPtr, &a1CSR3RowIndexDevPtr);
	_initDevCSR3Matrix(model.getA2InCSR3(), &a2CSR3ValuesDevPtr,
			&a2CSR3ColumnsDevPtr, &a2CSR3RowIndexDevPtr);
	_initDevCSR3Matrix(model.getA3InCSR3(), &a3CSR3ValuesDevPtr,
			&a3CSR3ColumnsDevPtr, &a3CSR3RowIndexDevPtr);
}

SolverContext::~SolverContext() {
	cudaFree(lDevPtr);

	cudaFree(a1CSR3ValuesDevPtr);
	cudaFree(a1CSR3ColumnsDevPtr);
	cudaFree(a1CSR3RowIndexDevPtr);

	cudaFree(a2CSR3ValuesDevPtr);
	cudaFree(a2CSR3ColumnsDevPtr);
	cudaFree(a2CSR3RowIndexDevPtr);

	cudaFree(a3CSR3ValuesDevPtr);
	cudaFree(a3CSR3ColumnsDevPtr);
	cudaFree(a3CSR3RowIndexDevPtr);

	freePtrs(solverPtrs);
	delete solverPtrs;

	freeDevicePtrs(solverDevPtrs);
	delete solverDevPtrs;

	freeDevicePtrs(svNormThresholdDevPtrs);
	delete svNormThresholdDevPtrs;

	freeDevicePtrs(sharedFloatDevPtrs);
	delete sharedFloatDevPtrs;

	freeDevicePtrs(sharedPointerDevPtrs);
	delete sharedPointerDevPtrs;

	freeDevicePtrs(k1DevPtrs);
	delete k1DevPtrs;

	freeDevicePtrs(k2DevPtrs);
	delete k2DevPtrs;

	freeDevicePtrs(k3DevPtrs);
	delete k3DevPtrs;

	freeDevicePtrs(k4DevPtrs);
	delete k4DevPtrs;

	freeDevicePtrs(prevStateDevPtrs);
	delete prevStateDevPtrs;

	freeDevicePtrs(curStateDevPtrs);
	delete curStateDevPtrs;
}

Solver * SolverContext::createSolverDev(const CUDA_COMPLEX_TYPE * const initialState) {
	if (svNormThresholdDevPtrs->size() > maxSolvers) {
		throw std::out_of_range(
				"Max solver number achieved - no more solvers may be created");
	}

	//block-locals : one per block
	FLOAT_TYPE * svNormThresholdDevPtr;
	checkCudaErrors(
			cudaMalloc((void**) &svNormThresholdDevPtr, sizeof(FLOAT_TYPE)));

	FLOAT_TYPE * sharedFloatDevPtr;
	checkCudaErrors(
			cudaMalloc((void**) &sharedFloatDevPtr, sizeof(FLOAT_TYPE)));

	CUDA_COMPLEX_TYPE ** sharedPointerDevPtr;
	checkCudaErrors(
			cudaMalloc((void**) &sharedPointerDevPtr, sizeof(CUDA_COMPLEX_TYPE *)));

	CUDA_COMPLEX_TYPE * k1DevPtr;
	checkCudaErrors(
			cudaMalloc((void**) &k1DevPtr, basisSize * sizeof(CUDA_COMPLEX_TYPE)));

	CUDA_COMPLEX_TYPE * k2DevPtr;
	checkCudaErrors(
			cudaMalloc((void**) &k2DevPtr, basisSize * sizeof(CUDA_COMPLEX_TYPE)));

	CUDA_COMPLEX_TYPE * k3DevPtr;
	checkCudaErrors(
			cudaMalloc((void**) &k3DevPtr, basisSize * sizeof(CUDA_COMPLEX_TYPE)));

	CUDA_COMPLEX_TYPE * k4DevPtr;
	checkCudaErrors(
			cudaMalloc((void**) &k4DevPtr, basisSize * sizeof(CUDA_COMPLEX_TYPE)));

	CUDA_COMPLEX_TYPE * prevStateDevPtr;
	checkCudaErrors(
			cudaMalloc((void**) &prevStateDevPtr, basisSize * sizeof(CUDA_COMPLEX_TYPE)));
	//set initial state
	transferState2Device(prevStateDevPtr, initialState);

	CUDA_COMPLEX_TYPE * curStateDevPtr;
	checkCudaErrors(
			cudaMalloc((void**) &curStateDevPtr, basisSize * sizeof(CUDA_COMPLEX_TYPE)));

	svNormThresholdDevPtrs->push_back(svNormThresholdDevPtr);
	sharedFloatDevPtrs->push_back(sharedFloatDevPtr);
	sharedPointerDevPtrs->push_back(sharedPointerDevPtr);

	k1DevPtrs->push_back(k1DevPtr);
	k2DevPtrs->push_back(k2DevPtr);
	k3DevPtrs->push_back(k3DevPtr);
	k4DevPtrs->push_back(k4DevPtr);

	prevStateDevPtrs->push_back(prevStateDevPtr);
	curStateDevPtrs->push_back(curStateDevPtr);

	Solver * solver = new Solver(basisSize, timeStep, nTimeSteps, lDevPtr,
			a1CSR3RowsNum, a1CSR3ValuesDevPtr, a1CSR3ColumnsDevPtr,
			a1CSR3RowIndexDevPtr, a2CSR3RowsNum, a2CSR3ValuesDevPtr,
			a2CSR3ColumnsDevPtr, a2CSR3RowIndexDevPtr, a3CSR3RowsNum,
			a3CSR3ValuesDevPtr, a3CSR3ColumnsDevPtr, a3CSR3RowIndexDevPtr,
			//block-local
			svNormThresholdDevPtr, sharedFloatDevPtr, sharedPointerDevPtr,
			k1DevPtr, k2DevPtr, k3DevPtr, k4DevPtr, prevStateDevPtr,
			curStateDevPtr);
	solverPtrs->push_back(solver);

	Solver * solverDevPtr = transferObject2Device(solver);
	solverDevPtrs->push_back(solverDevPtr);

	return solverDevPtr;
}

Solver ** SolverContext::createSolverDev(const uint count, const CUDA_COMPLEX_TYPE * const initialState) {
	Solver * solvers[count];
	for (int i = 0; i < count; ++i) {
		solvers[i] = createSolverDev(initialState);
	}

	return transferArray2Device(solvers, count);
}

void SolverContext::initAllSolvers(CUDA_COMPLEX_TYPE * initialState) {
	for (CUDA_COMPLEX_TYPE * prevStateDevPtr : *prevStateDevPtrs) {
		transferState2Device(prevStateDevPtr, initialState);
	}
}

void SolverContext::appendAllResults(std::vector<CUDA_COMPLEX_TYPE *> &results) {
	for (int i = 0; i < curStateDevPtrs->size(); ++i) {
		results.push_back(transferState2Host(curStateDevPtrs->at(i)));
	}
}

CUDA_COMPLEX_TYPE ** SolverContext::getAllResults() {
	CUDA_COMPLEX_TYPE ** results =
			new CUDA_COMPLEX_TYPE *[curStateDevPtrs->size()];

	for (int i = 0; i < curStateDevPtrs->size(); ++i) {
		results[i] = transferState2Host(curStateDevPtrs->at(i));
	}

	return results;
}

template<typename T>
inline void SolverContext::freeDevicePtrs(std::vector<T *> * &v) {
	for (T * devPtr : *v) {
		cudaFree(devPtr);
	}
}

template<typename T>
inline void SolverContext::freePtrs(std::vector<T *> * &v) {
	for (T * hostPtr : *v) {
		delete hostPtr;
	}
}

