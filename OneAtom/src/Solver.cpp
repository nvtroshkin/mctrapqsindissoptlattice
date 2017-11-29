/*
 * runge-kutta-solver.cpp
 *
 *  Created on: Nov 24, 2017
 *      Author: fake_sci
 */

#include <mkl_types.h>
#include <mkl_vsl.h>
#include <scalable_allocator.h>
#include <iostream>

#include <macro.h>
#include <mkl.h>
#include <eval-params.h>
#include <mkl-constants.h>

using namespace std;

class Solver {
	static COMPLEX_TYPE MULT_T_HALF_STEP;
	static COMPLEX_TYPE MULT_T_STEP;
	static COMPLEX_TYPE MULT_T_SIXTH_STEP;
	static COMPLEX_TYPE MULT_TWO;
	static COMPLEX_TYPE MULT_ONE;

	COMPLEX_TYPE *zeroVector;

	COMPLEX_TYPE *k1, *k2, *k3, *k4, *tempVector;

	//norms
	COMPLEX_TYPE norm2 { 1.0f, 0.0f }, normReversed { 1.0f, 0.0f };

	//random numbers
	int rndNumIndex;	//indicates where we are in the buffer
	FLOAT_TYPE *rndNumBuff;
	VSLStreamStatePtr Stream;

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

COMPLEX_TYPE Solver::MULT_T_HALF_STEP { 0.5f * T_STEP_SIZE, 0.0f };
COMPLEX_TYPE Solver::MULT_T_STEP { T_STEP_SIZE, 0.0f };
COMPLEX_TYPE Solver::MULT_T_SIXTH_STEP { T_STEP_SIZE / 6.0f, 0.0f };
COMPLEX_TYPE Solver::MULT_TWO { 2.0f, 0.0f };
COMPLEX_TYPE Solver::MULT_ONE { 1.0f, 0.0f };

Solver::Solver(MKL_INT basisSize) :
		rndNumIndex(0), basisSize(basisSize) {
	zeroVector = new COMPLEX_TYPE[basisSize];
	for (int i = 0; i < basisSize; i++) {
		zeroVector[i].real = 0.0f;
		zeroVector[i].imag = 0.0f;
	}

	k1 = new COMPLEX_TYPE[basisSize];
	k2 = new COMPLEX_TYPE[basisSize];
	k3 = new COMPLEX_TYPE[basisSize];
	k4 = new COMPLEX_TYPE[basisSize];
	tempVector = new COMPLEX_TYPE[basisSize];

	//create a random numbers stream
	rndNumBuff = (FLOAT_TYPE *) scalable_aligned_malloc(
			RND_NUM_BUFF_SIZE * sizeof(FLOAT_TYPE), SIMDALIGN);

	// Create a fake buffer with zeros for debugging
	//	for (int i = 0; i < RND_NUM_BUFF_SIZE; i++) {
	//		rndNumBuff[i]=0.0;
	//	}

	vslNewStream(&Stream, VSL_BRNG_MCG31, RANDSEED);
	vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, Stream, RND_NUM_BUFF_SIZE,
			rndNumBuff, 0.0f, 1.0f);

	//save the realization to a file
	//	ofstream myfile;
	//	myfile.open("rnd-numbers.txt");
	//	if (!myfile.is_open()) {
	//		cout << "Can't open file!" << endl;
	//	}
	//	for (int i = 0; i < RND_NUM_BUFF_SIZE; i++) {
	//		myfile << rndNumBuff[i] << ", ";
	//	}
	//	myfile.close();
}

Solver::~Solver() {
	delete[] zeroVector;
	delete[] k1;
	delete[] k2;
	delete[] k3;
	delete[] k4;
	delete[] tempVector;

	scalable_aligned_free(rndNumBuff);
	vslDeleteStream(&Stream);
}

/**
 * Stores the final result in the curStep
 */
void Solver::solve(const COMPLEX_TYPE *HCSR3Values, const int *HCSR3RowIndex,
		const int *HCSR3Columns, const COMPLEX_TYPE *aCSR3Values,
		const int *aCSR3RowIndex, const int *aCSR3Columns,
		COMPLEX_TYPE *prevState, COMPLEX_TYPE *curState) {
	COMPLEX_TYPE *tempPointer;

	//get a random number for the calculation of the random waiting time
	//of the next jump
	FLOAT_TYPE svNormThreshold = rndNumBuff[rndNumIndex++];

	//Calculate each sample by the time axis
	for (int i = 0; i < TIME_STEPS_NUMBER; i++) {
		//Before the next jump there is deterministic evolution guided by
		//the Shroedinger's equation

		//A jump occurs between t(i) and t(i+1) when square of the state vector norm
		//at t(i+1)...
		//write the f function

		make4thOrderRungeKuttaStep(HCSR3Values, HCSR3RowIndex, HCSR3Columns,
				prevState, curState);

		//...falls below the threshold, which is a random number
		//uniformly distributed between [0,1] - svNormThreshold

		//if the state vector at t(i+1) has a less square of the norm then the threshold
		//try a self-written norm?
		cblas_cdotc_sub(basisSize, curState, NO_INC, curState, NO_INC, &norm2);
		if (svNormThreshold > norm2.real) {
			//then a jump is occurred between t(i) and t(i+1)
			//let's suppose it was at time t(i)

			//calculate the state vector after the jump
			//store it at t(i+1)
			mkl_cspblas_ccsrgemv("n", &basisSize, aCSR3Values,
					aCSR3RowIndex, aCSR3Columns, prevState, curState);
			//calculate new norm
			cblas_cdotc_sub(basisSize, curState, NO_INC, curState, NO_INC,
					&norm2);

			//update the random time
			svNormThreshold = rndNumBuff[rndNumIndex++];

			//normalize vector
			normalizeVector(curState);
		}

		//update indices
		tempPointer = curState;
		curState = prevState;
		prevState = tempPointer;
	}

	//swap back - curState should hold the final result
	tempPointer = curState;
	curState = prevState;
	prevState = tempPointer;

	//final state normalization
	normalizeVector(curState);

	//check if random numbers valid - replace with correct random numbers generation
	if (rndNumIndex >= RND_NUM_BUFF_SIZE) {
		cout << "No more random numbers" << endl;
	}
}

inline void Solver::make4thOrderRungeKuttaStep(const COMPLEX_TYPE *HCSR3Values,
		const int *HCSR3RowIndex, const int *HCSR3Columns,
		const COMPLEX_TYPE *prevState, COMPLEX_TYPE *curState) {
	//uses nextState as a temporary storage vector

	//k1 = f(t, sample[i]);
	//to k1
	mkl_cspblas_ccsrgemv("n", &basisSize, HCSR3Values, HCSR3RowIndex,
			HCSR3Columns, prevState, k1);

	//copy current state to a temporary vector
	cblas_ccopy(basisSize, prevState, NO_INC, curState, NO_INC);
	cblas_caxpy(basisSize, &MULT_T_HALF_STEP, k1, NO_INC, curState, NO_INC);
	//k2 = f(t + T_STEP_SIZE / 2.0f, VECTOR_BUFF[1]);
	mkl_cspblas_ccsrgemv("n", &basisSize, HCSR3Values, HCSR3RowIndex,
			HCSR3Columns, curState, k2);

	//same but with another temporary vector for the buffer
	cblas_ccopy(basisSize, prevState, NO_INC, curState, NO_INC);
	cblas_caxpy(basisSize, &MULT_T_HALF_STEP, k2, NO_INC, curState, NO_INC);
	//k3 = f(t + T_STEP_SIZE / 2.0f, VECTOR_BUFF[2])
	mkl_cspblas_ccsrgemv("n", &basisSize, HCSR3Values, HCSR3RowIndex,
			HCSR3Columns, curState, k3);

	cblas_ccopy(basisSize, prevState, NO_INC, curState, NO_INC);
	cblas_caxpy(basisSize, &MULT_T_STEP, k3, NO_INC, curState, NO_INC);
	//k4 = f(t + T_STEP_SIZE, VECTOR_BUFF[3]);
	mkl_cspblas_ccsrgemv("n", &basisSize, HCSR3Values, HCSR3RowIndex,
			HCSR3Columns, curState, k4);

	//store to k1
	cblas_caxpy(basisSize, &MULT_TWO, k2, NO_INC, k1, NO_INC);
	//to k4
	cblas_caxpy(basisSize, &MULT_TWO, k3, NO_INC, k4, NO_INC);
	//to k1
	cblas_caxpy(basisSize, &MULT_ONE, k4, NO_INC, k1, NO_INC);
	//modify sample[i+1]
	cblas_ccopy(basisSize, prevState, NO_INC, curState, NO_INC);
	cblas_caxpy(basisSize, &MULT_T_SIXTH_STEP, k1, NO_INC, curState, NO_INC);
}

inline void Solver::normalizeVector(COMPLEX_TYPE *stateVector) {
	vsSqrt((MKL_INT) 1, &(norm2.real), &(normReversed.real));
	normReversed.real = 1.0f / normReversed.real;

	cblas_ccopy(basisSize, zeroVector, NO_INC, tempVector, NO_INC);
	cblas_caxpy(basisSize, &normReversed, stateVector, NO_INC, tempVector,
			NO_INC);
	//write back
	cblas_ccopy(basisSize, tempVector, NO_INC, stateVector, NO_INC);
}

