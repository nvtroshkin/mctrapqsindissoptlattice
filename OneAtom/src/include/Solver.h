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
#include <ModelBuilder.h>
#include <RndNumProvider.h>

class Solver {
	//an alignment for memory in a multithreaded environment
	static const int SIMDALIGN = 1024;

	const COMPLEX_TYPE complexTHalfStep;
	const COMPLEX_TYPE complexTStep;
	const COMPLEX_TYPE complexTSixthStep;
	const COMPLEX_TYPE complexTwo;
	const COMPLEX_TYPE complexOne;

	const MKL_INT basisSize;
	const int timeStepsNumber;

	//norms
	COMPLEX_TYPE norm2 { 1.0, 0.0 }, normReversed { 1.0, 0.0 };

	//the model
	const CSR3Matrix * const hCSR3;
	const CSR3Matrix * const aCSR3;
	const CSR3Matrix * const aPlusCSR3;

	RndNumProvider &rndNumProvider;

	//caches
	COMPLEX_TYPE *zeroVector;COMPLEX_TYPE *k1, *k2, *k3, *k4, *tempVector,
			*prevState, *curState;

	//random numbers
	int rndNumIndex;	//indicates where we are in the buffer
	FLOAT_TYPE *rndNumBuff;

	void make4thOrderRungeKuttaStep(std::ostream &consoleStream,
			const COMPLEX_TYPE *HCSR3Values, const int *HCSR3RowIndex,
			const int *HCSR3Columns);

public:
	Solver(int id, MKL_INT basisSize, FLOAT_TYPE timeStep, int timeStepsNumber,
			ModelBuilder &modelBuilder, RndNumProvider &rndNumProvider);
	~Solver();

	/**
	 * Stores the final result in the curStep
	 */
	void solve(std::ostream &consoleStream, const COMPLEX_TYPE * initialState,
	COMPLEX_TYPE * const resultState);

	void normalizeVector(COMPLEX_TYPE *stateVector);

	void makeJump(std::ostream &consoleStream, FLOAT_TYPE &svNormThreshold,
	COMPLEX_TYPE *prevState, COMPLEX_TYPE *curState);
};

#endif /* SRC_INCLUDE_SOLVER_H_ */
