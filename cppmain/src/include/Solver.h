/*
 * runge-kutta-solver.h
 *
 *  Created on: Nov 24, 2017
 *      Author: fakesci
 */

#ifndef SRC_INCLUDE_SOLVER_H_
#define SRC_INCLUDE_SOLVER_H_

#include <precision-definition.h>
#include <iostream>
#include <CSR3Matrix.h>
#include <Model.h>
#include <RndNumProvider.h>

#ifdef USE_GPU
#include <cublas_v2.h>
#endif

class Solver {
	//an alignment for memory in a multithreaded environment
	static const int SIMDALIGN = 1024;

	const int id;

	const COMPLEX_TYPE complexTHalfStep;
	const COMPLEX_TYPE complexTStep;
	const COMPLEX_TYPE complexTSixthStep;
	const COMPLEX_TYPE complexTwo;
	const COMPLEX_TYPE complexOne;
	const COMPLEX_TYPE complexZero;

	const MKL_INT basisSize;
	const int timeStepsNumber;

	//norms
	COMPLEX_TYPE norm2 { 1.0, 0.0 }, normReversed { 1.0, 0.0 }, n12, n22, n32;

	//the model
#ifdef H_SPARSE
	const CSR3Matrix * const lCSR3;
#else
	const COMPLEX_TYPE * const l;
#endif

	const CSR3Matrix * const a1CSR3;
	const CSR3Matrix * const a1PlusCSR3;
	const CSR3Matrix * const a2CSR3;
	const CSR3Matrix * const a2PlusCSR3;
	const CSR3Matrix * const a3CSR3;
	const CSR3Matrix * const a3PlusCSR3;

	RndNumProvider &rndNumProvider;

#ifdef USE_GPU
	const cublasHandle_t cublasHandle;
	const CUDA_COMPLEX_TYPE * const devPtrL;

	//cache
	CUDA_COMPLEX_TYPE * devPtrVector;
	CUDA_COMPLEX_TYPE * devPtrResult;
#endif

	//caches
	COMPLEX_TYPE *zeroVector;

	COMPLEX_TYPE *k1, *k2, *k3, *k4, *tempVector, *prevState, *curState;

#if defined(DEBUG_CONTINUOUS) || defined(DEBUG_JUMPS)
	bool shouldPrintDebugInfo = false;
#endif

	//random numbers
	int rndNumIndex;	//indicates where we are in the buffer
	FLOAT_TYPE *rndNumBuff;

	void make4thOrderRungeKuttaStep(std::ostream &consoleStream);

	void normalizeVector(COMPLEX_TYPE *stateVector,
			const COMPLEX_TYPE &stateVectorNorm2,
			COMPLEX_TYPE *result);

	FLOAT_TYPE nextRandom();

	void multLOnVector(COMPLEX_TYPE *vector, COMPLEX_TYPE *result);
public:
	Solver(int id, FLOAT_TYPE timeStep, int timeStepsNumber, Model &model,
			RndNumProvider &rndNumProvider
#ifdef USE_GPU
			, const cublasHandle_t cublasHandle,
			const CUDA_COMPLEX_TYPE * const devPtrL
#endif
			);
	~Solver();

	/**
	 * Stores the final result in the curStep
	 */
	void solve(std::ostream &consoleStream, const COMPLEX_TYPE * initialState,
	COMPLEX_TYPE * const resultState);

	void normalizeVector(COMPLEX_TYPE *stateVector);

	void makeJump(std::ostream &consoleStream, COMPLEX_TYPE *prevState,
	COMPLEX_TYPE *curState);
};

#endif /* SRC_INCLUDE_SOLVER_H_ */
