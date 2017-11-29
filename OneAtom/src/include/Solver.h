/*
 * runge-kutta-solver.h
 *
 *  Created on: Nov 24, 2017
 *      Author: fake_sci
 */

#ifndef SRC_INCLUDE_SOLVER_H_
#define SRC_INCLUDE_SOLVER_H_

#include <macro.h>

class Solver {
	static COMPLEX_TYPE MULT_T_HALF_STEP;
	static COMPLEX_TYPE MULT_T_STEP;
	static COMPLEX_TYPE MULT_T_SIXTH_STEP;
	static COMPLEX_TYPE MULT_TWO;
	static COMPLEX_TYPE MULT_ONE;

	COMPLEX_TYPE *zeroVector;

	COMPLEX_TYPE *k1, *k2, *k3, *k4, *tempVector;

	//norms
	COMPLEX_TYPE norm2, normReversed;

	//random numbers
	int rndNumIndex;	//indicates where we are in the buffer
	FLOAT_TYPE *rndNumBuff;

	MKL_INT basisSize;

	void make4thOrderRungeKuttaStep(const COMPLEX_TYPE *HCSR3Values,
			const int *HCSR3RowIndex, const int *HCSR3Columns,
			const COMPLEX_TYPE *prevState, COMPLEX_TYPE *curState);

	void normalizeVector(COMPLEX_TYPE *stateVector);
public:
	Solver(MKL_INT basisSize);
	~Solver();

	/**
	 * Stores the final result in the curStep
	 */
	void solve(const COMPLEX_TYPE *HCSR3Values, const int *HCSR3RowIndex,
			const int *HCSR3Columns, const COMPLEX_TYPE *aCSR3Values,
			const int *aCSR3RowIndex, const int *aCSR3Columns,
			COMPLEX_TYPE *prevState, COMPLEX_TYPE *curState);
};

#endif /* SRC_INCLUDE_SOLVER_H_ */
