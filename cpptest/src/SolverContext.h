/*
 * SolverFactory.h
 *
 *  Created on: Dec 20, 2017
 *      Author: fakesci
 */

#ifndef SRC_SOLVERCONTEXT_H_
#define SRC_SOLVERCONTEXT_H_

#include "vector"

#include "cuda_runtime.h"
#include "helper_cuda.h"

#include "precision-definition.h"
#include "Solver.h"
#include "Model.h"

class SolverContext {

	//---------------------global-------------------------------
	const uint maxSolvers;
	const uint basisSize;
	const FLOAT_TYPE timeStep;
	const uint nTimeSteps;

	CUDA_COMPLEX_TYPE * lDevPtr;

	const int a1CSR3RowsNum;

	CUDA_COMPLEX_TYPE * a1CSR3ValuesDevPtr;
	int * a1CSR3ColumnsDevPtr;
	int * a1CSR3RowIndexDevPtr;

	const int a2CSR3RowsNum;

	CUDA_COMPLEX_TYPE * a2CSR3ValuesDevPtr;
	int * a2CSR3ColumnsDevPtr;
	int * a2CSR3RowIndexDevPtr;

	const int a3CSR3RowsNum;

	CUDA_COMPLEX_TYPE * a3CSR3ValuesDevPtr;
	int * a3CSR3ColumnsDevPtr;
	int * a3CSR3RowIndexDevPtr;

	//----------------solver-local--------------------------------
	std::vector<Solver *> * solverPtrs;
	std::vector<Solver *> * solverDevPtrs;

	std::vector<FLOAT_TYPE *> * svNormThresholdDevPtrs;
	std::vector<FLOAT_TYPE *> * sharedFloatDevPtrs;
	std::vector<CUDA_COMPLEX_TYPE **> * sharedPointerDevPtrs;

	std::vector<CUDA_COMPLEX_TYPE *> * k1DevPtrs;
	std::vector<CUDA_COMPLEX_TYPE *> * k2DevPtrs;
	std::vector<CUDA_COMPLEX_TYPE *> * k3DevPtrs;
	std::vector<CUDA_COMPLEX_TYPE *> * k4DevPtrs;

	std::vector<CUDA_COMPLEX_TYPE *> * prevStateDevPtrs;
	std::vector<CUDA_COMPLEX_TYPE *> * curStateDevPtrs; //also holds results

	//----------------Methods declarations------------------------

	template<typename T>
	void freeDevicePtrs(std::vector<T *> * &v);

	template<typename T>
	void freePtrs(std::vector<T *> * &v);

public:
	SolverContext(uint maxSolvers, FLOAT_TYPE timeStep, uint nTimeSteps,
			Model &model);
	~SolverContext();

	Solver * createSolverDev(CUDA_COMPLEX_TYPE * initialState);

	void initAllSolvers(CUDA_COMPLEX_TYPE * initialState);

	CUDA_COMPLEX_TYPE ** getAllResults();

	template<typename T> void transferState2Device(T * const devPtr,
			const T * const hostPtr);

	template<typename T> T* transferState2Device(const T * const hostPtr);

	template<typename T> void transferState2Host(const T * const devPtr,
			T * const hostPtr);

	template<typename T> T* transferState2Host(const T * const devPtr);

	template<typename T> T * transferObject2Device(const T * const hostPtr);

	template<typename T> T* transferArray2Host(const T * const devPtr,
			uint size);

	template<typename T> void transferArray2Host(const T * const devPtr,
			T * const hostPtr, const uint size);

	//---------------------------Getters-------------------------------

	const CUDA_COMPLEX_TYPE * const getDevPtrL();

	int getA1CSR3RowsNum();

	const CUDA_COMPLEX_TYPE * const getA1CSR3ValuesDevPtr();
	const int * const getA1CSR3ColumnsDevPtr();
	const int * const getA1CSR3RowIndexDevPtr();

	int getA2CSR3RowsNum();

	const CUDA_COMPLEX_TYPE * const getA2CSR3ValuesDevPtr();
	const int * const getA2CSR3ColumnsDevPtr();
	const int * const getA2CSR3RowIndexDevPtr();

	int getA3CSR3RowsNum();

	const CUDA_COMPLEX_TYPE * const getA3CSR3ValuesDevPtr();
	const int * const getA3CSR3ColumnsDevPtr();
	const int * const getA3CSR3RowIndexDevPtr();
};

template<typename T> inline void SolverContext::transferState2Device(
		T * const devPtr, const T * const hostPtr) {
	checkCudaErrors(
			cudaMemcpy(devPtr, hostPtr, basisSize * sizeof(T),
					cudaMemcpyHostToDevice));
}

template<typename T> inline T* SolverContext::transferState2Device(
		const T * const hostPtr) {
	T * devPtr;
	checkCudaErrors(cudaMalloc((void**) &devPtr, basisSize * sizeof(T)));
	transferState2Device(devPtr, hostPtr);

	return devPtr;
}

template<typename T> inline void SolverContext::transferState2Host(
		const T * const devPtr, T * const hostPtr) {
	transferArray2Host(devPtr, hostPtr, basisSize);
}

template<typename T> inline T* SolverContext::transferState2Host(
		const T * const devPtr) {
	return transferArray2Host(devPtr, basisSize);
}

template<typename T> inline T* SolverContext::transferArray2Host(
		const T * const devPtr, const uint size) {
	T * hostPtr = new T[size];
	transferArray2Host(devPtr, hostPtr, size);

	return hostPtr;
}

template<typename T> inline void SolverContext::transferArray2Host(
		const T * const devPtr, T * const hostPtr, const uint size) {
	checkCudaErrors(
			cudaMemcpy(hostPtr, devPtr, size * sizeof(T),
					cudaMemcpyDeviceToHost));
}

template<typename T> inline T* SolverContext::transferObject2Device(
		const T * const hostPtr) {
	T * devPtr;
	checkCudaErrors(cudaMalloc((void**) &devPtr, sizeof(T)));
	checkCudaErrors(
			cudaMemcpy(devPtr, hostPtr, sizeof(T), cudaMemcpyHostToDevice));

	return devPtr;
}

//-----------------------------Getters---------------------------------

inline const CUDA_COMPLEX_TYPE * const SolverContext::getDevPtrL() {
	return lDevPtr;
}

inline int SolverContext::getA1CSR3RowsNum() {
	return a1CSR3RowsNum;
}

inline const CUDA_COMPLEX_TYPE * const SolverContext::getA1CSR3ValuesDevPtr() {
	return a1CSR3ValuesDevPtr;
}

inline const int * const SolverContext::getA1CSR3ColumnsDevPtr() {
	return a1CSR3ColumnsDevPtr;
}

inline const int * const SolverContext::getA1CSR3RowIndexDevPtr() {
	return a1CSR3RowIndexDevPtr;
}

inline int SolverContext::getA2CSR3RowsNum() {
	return a2CSR3RowsNum;
}

inline const CUDA_COMPLEX_TYPE * const SolverContext::getA2CSR3ValuesDevPtr() {
	return a2CSR3ValuesDevPtr;
}

inline const int * const SolverContext::getA2CSR3ColumnsDevPtr() {
	return a2CSR3ColumnsDevPtr;
}

inline const int * const SolverContext::getA2CSR3RowIndexDevPtr() {
	return a2CSR3RowIndexDevPtr;
}

inline int SolverContext::getA3CSR3RowsNum() {
	return a3CSR3RowsNum;
}

inline const CUDA_COMPLEX_TYPE * const SolverContext::getA3CSR3ValuesDevPtr() {
	return a3CSR3ValuesDevPtr;
}

inline const int * const SolverContext::getA3CSR3ColumnsDevPtr() {
	return a3CSR3ColumnsDevPtr;
}

inline const int * const SolverContext::getA3CSR3RowIndexDevPtr() {
	return a3CSR3RowIndexDevPtr;
}
#endif /* SRC_SOLVERCONTEXT_H_ */
