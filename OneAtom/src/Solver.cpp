/*
 * runge-kutta-solver.cpp
 *
 *  Created on: Nov 24, 2017
 *      Author: fake_sci
 */

#include <mkl.h>
#include <scalable_allocator.h>
#include <iostream>

#include <mkl-constants.h>
#include <precision-definition.h>
#include <ModelBuilder.h>
#include <RndNumProvider.h>

#if defined(DEBUG_MODE) || defined(DEBUG_JUMPS)
#include <utilities.h>
#endif

#include <Solver.h>

using std::endl;

Solver::Solver(MKL_INT basisSize, FLOAT_TYPE timeStep, int timeStepsNumber,
		ModelBuilder &modelBuilder, RndNumProvider &rndNumProvider) :
		complexTHalfStep( { 0.5 * timeStep, 0.0 }), complexTStep( { timeStep,
				0.0 }), complexTSixthStep( { timeStep / 6.0, 0.0 }), complexTwo(
				{ 2.0, 0.0 }), complexOne( { 1.0, 0.0 }), basisSize(basisSize), timeStepsNumber(
				timeStepsNumber), rndNumIndex(0), hCSR3(
				modelBuilder.getHInCSR3()), aCSR3(modelBuilder.getAInCSR3()), aPlusCSR3(
				modelBuilder.getAPlusInCSR3()), rndNumProvider(rndNumProvider) {
	zeroVector = new COMPLEX_TYPE[basisSize];
	for (int i = 0; i < basisSize; i++) {
		zeroVector[i].real = 0.0;
		zeroVector[i].imag = 0.0;
	}

	k1 = new COMPLEX_TYPE[basisSize];
	k2 = new COMPLEX_TYPE[basisSize];
	k3 = new COMPLEX_TYPE[basisSize];
	k4 = new COMPLEX_TYPE[basisSize];
	tempVector = new COMPLEX_TYPE[basisSize];
	prevState = new COMPLEX_TYPE[basisSize];
	curState = new COMPLEX_TYPE[basisSize];

	//create a random numbers stream
	rndNumBuff = (FLOAT_TYPE *) scalable_aligned_malloc(
			RND_NUM_BUFF_SIZE * sizeof(FLOAT_TYPE), SIMDALIGN);

	// Create a fake buffer with zeros for debugging
	//	for (int i = 0; i < RND_NUM_BUFF_SIZE; i++) {
	//		rndNumBuff[i]=0.0;
	//	}

	rndNumProvider.initBuffer(0, rndNumBuff, RND_NUM_BUFF_SIZE);

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
	delete[] prevState;
	delete[] curState;

	scalable_aligned_free(rndNumBuff);
}

/**
 * Stores the final result in the curStep
 */
void Solver::solve(std::ostream &consoleStream,
		const COMPLEX_TYPE * const initialState,
		COMPLEX_TYPE * const resultState) {
	//prepare state
	complex_cblas_copy(basisSize, initialState, NO_INC, prevState, NO_INC);

	COMPLEX_TYPE *tempPointer;

	//get a random number for the calculation of the random waiting time
	//of the next jump
	FLOAT_TYPE svNormThreshold = rndNumBuff[rndNumIndex++];

	//Calculate each sample by the time axis
	for (int i = 0; i < timeStepsNumber; i++) {

#ifdef DEBUG_MODE
		consoleStream << "Solver: step " << i << endl;
#endif

		//Before the next jump there is deterministic evolution guided by
		//the Shroedinger's equation

		//A jump occurs between t(i) and t(i+1) when square of the state vector norm
		//at t(i+1)...
		//write the f function

		make4thOrderRungeKuttaStep(consoleStream, hCSR3->values,
				hCSR3->rowIndex, hCSR3->columns);

		//...falls below the threshold, which is a random number
		//uniformly distributed between [0,1] - svNormThreshold

		//if the state vector at t(i+1) has a less square of the norm then the threshold
		//try a self-written norm?
		complex_cblas_dotc_sub(basisSize, curState, NO_INC, curState, NO_INC,
				&norm2);
		if (svNormThreshold > norm2.real) {
#ifdef DEBUG_JUMPS
			consoleStream << "jump at step: " << i << endl;
			consoleStream << "vector norm: " << norm2.real << endl;
			print(consoleStream, "vector: ", curState, basisSize);
#endif

			makeJump(consoleStream, svNormThreshold, prevState, curState);

#ifdef DEBUG_JUMPS
			print(consoleStream, "vector after jump and normalization: ",
					curState, basisSize);
#endif
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

#ifdef DEBUG_MODE
	print(consoleStream, "Psi(n+1): ", curState, basisSize);
#endif

	//final state normalization
	normalizeVector(curState);

	//check if random numbers valid - replace with correct random numbers generation
	if (rndNumIndex >= RND_NUM_BUFF_SIZE) {
		std::cout << "No more random numbers" << endl;
	}

#ifdef DEBUG_MODE
	print(consoleStream, "Normed Psi(n+1): ", curState, basisSize);
#endif

	//write results out
	complex_cblas_copy(basisSize, curState, NO_INC, resultState, NO_INC);
}

inline void Solver::make4thOrderRungeKuttaStep(std::ostream &consoleStream,
		const COMPLEX_TYPE *HCSR3Values, const int *HCSR3RowIndex,
		const int *HCSR3Columns) {
	//uses nextState as a temporary storage vector

	//k1 = f(t, sample[i]);
	//to k1
	complex_mkl_cspblas_csrgemv("n", &basisSize, HCSR3Values, HCSR3RowIndex,
			HCSR3Columns, prevState, k1);

#ifdef DEBUG_MODE
	print(consoleStream, "k1: ", k1, basisSize);
#endif

	//copy current state to a temporary vector
	complex_cblas_copy(basisSize, prevState, NO_INC, curState, NO_INC);
	complex_cblas_axpy(basisSize, &complexTHalfStep, k1, NO_INC, curState,
			NO_INC);
	//k2 = f(t + T_STEP_SIZE / 2.0f, VECTOR_BUFF[1]);
	complex_mkl_cspblas_csrgemv("n", &basisSize, HCSR3Values, HCSR3RowIndex,
			HCSR3Columns, curState, k2);

#ifdef DEBUG_MODE
	print(consoleStream, "k2: ", k2, basisSize);
#endif

	//same but with another temporary vector for the buffer
	complex_cblas_copy(basisSize, prevState, NO_INC, curState, NO_INC);
	complex_cblas_axpy(basisSize, &complexTHalfStep, k2, NO_INC, curState,
			NO_INC);
	//k3 = f(t + T_STEP_SIZE / 2.0f, VECTOR_BUFF[2])
	complex_mkl_cspblas_csrgemv("n", &basisSize, HCSR3Values, HCSR3RowIndex,
			HCSR3Columns, curState, k3);

#ifdef DEBUG_MODE
	print(consoleStream, "k3: ", k3, basisSize);
#endif

	complex_cblas_copy(basisSize, prevState, NO_INC, curState, NO_INC);
	complex_cblas_axpy(basisSize, &complexTStep, k3, NO_INC, curState, NO_INC);
	//k4 = f(t + T_STEP_SIZE, VECTOR_BUFF[3]);
	complex_mkl_cspblas_csrgemv("n", &basisSize, HCSR3Values, HCSR3RowIndex,
			HCSR3Columns, curState, k4);

#ifdef DEBUG_MODE
	print(consoleStream, "k4: ", k4, basisSize);
#endif

	//store to k1
	complex_cblas_axpy(basisSize, &complexTwo, k2, NO_INC, k1, NO_INC);
	//to k4
	complex_cblas_axpy(basisSize, &complexTwo, k3, NO_INC, k4, NO_INC);
	//to k1
	complex_cblas_axpy(basisSize, &complexOne, k4, NO_INC, k1, NO_INC);
	//modify sample[i+1]
	complex_cblas_copy(basisSize, prevState, NO_INC, curState, NO_INC);
	complex_cblas_axpy(basisSize, &complexTSixthStep, k1, NO_INC, curState,
			NO_INC);
}

inline void Solver::normalizeVector(COMPLEX_TYPE *stateVector) {
	//calculate new norm
	complex_cblas_dotc_sub(basisSize, stateVector, NO_INC, stateVector, NO_INC,
			&norm2);

	vSqrt((MKL_INT) 1, &(norm2.real), &(normReversed.real));
	normReversed.real = 1.0 / normReversed.real;

	complex_cblas_copy(basisSize, zeroVector, NO_INC, tempVector, NO_INC);
	complex_cblas_axpy(basisSize, &normReversed, stateVector, NO_INC,
			tempVector, NO_INC);
	//write back
	complex_cblas_copy(basisSize, tempVector, NO_INC, stateVector, NO_INC);
}

inline void Solver::makeJump(std::ostream &consoleStream,
FLOAT_TYPE &svNormThreshold, COMPLEX_TYPE *prevState, COMPLEX_TYPE *curState) {
	//then a jump is occurred between t(i) and t(i+1)
	//let's suppose it was at time t(i)

	//calculate the state vector after the jump
	//store it at t(i+1)
	complex_mkl_cspblas_csrgemv("n", &basisSize, aCSR3->values, aCSR3->rowIndex,
			aCSR3->columns, prevState, curState);

	//normalize vector - may not leave it unnormalized because the norm
	//should fall from one during the unitary phase
	normalizeVector(curState);

	//update the random time
	svNormThreshold = rndNumBuff[rndNumIndex++];
}

