/*
 *  Created on: Nov 24, 2017
 *      Author: fakesci
 */

#include <tbb/scalable_allocator.h>
#include <string>
#include <iostream>
#include "include/precision-definition.h"
#include "include/Model.h"
#include "include/RndNumProvider.h"
#include "include/Solver.h"
#include "include/eval-params.h"
#include "mkl-constants.h"

#if defined(DEBUG_CONTINUOUS) || defined(DEBUG_JUMPS)
#include <utilities.h>
#define LOG_IF_APPROPRIATE(a) if(shouldPrintDebugInfo) (a)
#endif

#include <cublas_v2.h>
#include <cuda_runtime.h>

using std::endl;

Solver::Solver(int id, FLOAT_TYPE timeStep, int timeStepsNumber, Model &model,
		RndNumProvider &rndNumProvider, const cublasHandle_t cublasHandle,
		const CUDA_COMPLEX_TYPE * const devPtrL) :
		id(id), complexTHalfStep( { 0.5 * timeStep, 0.0 }), complexTStep( {
				timeStep, 0.0 }), complexTSixthStep( { timeStep / 6.0, 0.0 }), complexTwo(
				{ 2.0, 0.0 }), complexOne( { 1.0, 0.0 }), complexZero( { 0.0,
				0.0 }), basisSize(model.getBasisSize()), timeStepsNumber(
				timeStepsNumber),
#ifdef H_SPARSE
				lCSR3(
						model.getLInCSR3())
#else
				l(model.getL())
#endif
						, a1CSR3(model.getA1InCSR3()), a1PlusCSR3(
						model.getA1PlusInCSR3()), a2CSR3(model.getA2InCSR3()), a2PlusCSR3(
						model.getA2PlusInCSR3()), a3CSR3(model.getA3InCSR3()), a3PlusCSR3(
						model.getA3PlusInCSR3()), rndNumProvider(
						rndNumProvider), cublasHandle(cublasHandle), devPtrL(
						devPtrL),

				rndNumIndex(0) {
	cudaError_t cudaStatus = cudaMalloc((void**) &devPtrVector,
			basisSize * sizeof(CUDA_COMPLEX_TYPE));
	if (cudaStatus != cudaSuccess) {
		std::cout
				<< std::string(
						"Device memory allocation for devPtrVector failed with code:")
						+ std::to_string(cudaStatus) << endl;
		throw cudaStatus;
	}

	cudaStatus = cudaMalloc((void**) &devPtrResult,
			basisSize * sizeof(CUDA_COMPLEX_TYPE));
	if (cudaStatus != cudaSuccess) {
		cudaFree(devPtrVector);
		std::cout
				<< std::string(
						"Device memory allocation for devPtrResult failed with code:")
						+ std::to_string(cudaStatus) << endl;
		throw cudaStatus;
	}

	zeroVector = new COMPLEX_TYPE[basisSize];
	for (int i = 0; i < basisSize; ++i) {
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

	rndNumProvider.initBuffer(id, rndNumBuff, RND_NUM_BUFF_SIZE);
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

	cudaFree(devPtrVector);
	cudaFree(devPtrResult);
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
	FLOAT_TYPE svNormThreshold = nextRandom();

	//Calculate each sample by the time axis
	for (int i = 0; i < timeStepsNumber; ++i) {
#if defined(DEBUG_CONTINUOUS) || defined(DEBUG_JUMPS)
		shouldPrintDebugInfo = (i % TIME_STEPS_BETWEEN_DEBUG == 0);
		LOG_IF_APPROPRIATE(consoleStream << "Solver: step " << i << endl);
#endif

		//Before the next jump there is deterministic evolution guided by
		//the Shroedinger's equation

		//A jump occurs between t(i) and t(i+1) when square of the state vector norm
		//at t(i+1)...
		//write the f function

		make4thOrderRungeKuttaStep(consoleStream);

		//...falls below the threshold, which is a random number
		//uniformly distributed between [0,1] - svNormThreshold

		//if the state vector at t(i+1) has a less square of the norm then the threshold
		//try a self-written norm?
		complex_cblas_dotc_sub(basisSize, curState, NO_INC, curState, NO_INC,
				&norm2);

#ifdef DEBUG_JUMPS
		LOG_IF_APPROPRIATE(
				consoleStream << "Norm: threshold = " << svNormThreshold
				<< ", current = " << norm2.real << endl);
#endif

		if (svNormThreshold > norm2.real) {
#ifdef DEBUG_JUMPS
			consoleStream << "Jump" << endl;
			consoleStream << "Step: " << i << endl;
			consoleStream << "Norm^2: threshold = " << svNormThreshold
			<< ", current = " << norm2.real << endl;
#endif

			makeJump(consoleStream, prevState, curState);

			//update the random time
			svNormThreshold = nextRandom();

#ifdef DEBUG_JUMPS
			consoleStream << "New norm^2 threshold = " << svNormThreshold
			<< endl;
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

#ifdef DEBUG_CONTINUOUS
	LOG_IF_APPROPRIATE(print(consoleStream, "Psi(n+1)", curState, basisSize));
#endif

	//final state normalization
	normalizeVector(curState);

	//check if random numbers valid - replace with correct random numbers generation
	if (rndNumIndex >= RND_NUM_BUFF_SIZE) {
		throw "No more random numbers";
	}

#ifdef DEBUG_CONTINUOUS
	LOG_IF_APPROPRIATE(
			print(consoleStream, "Normed Psi(n+1)", curState, basisSize));
#endif

	//write results out
	complex_cblas_copy(basisSize, curState, NO_INC, resultState, NO_INC);
}

inline void Solver::make4thOrderRungeKuttaStep(std::ostream &consoleStream) {
	//uses nextState as a temporary storage vector

#ifdef DEBUG_CONTINUOUS
	LOG_IF_APPROPRIATE(print(consoleStream, "prevState", prevState, basisSize));
#endif

	//k1 = f(t, sample[i]);
	//to k1
	multLOnVector(prevState, k1);

#ifdef DEBUG_CONTINUOUS
	LOG_IF_APPROPRIATE(print(consoleStream, "k1", k1, basisSize));
#endif

	//copy current state to a temporary vector
	complex_cblas_copy(basisSize, prevState, NO_INC, curState, NO_INC);
	complex_cblas_axpy(basisSize, &complexTHalfStep, k1, NO_INC, curState,
			NO_INC);
	//k2 = f(t + T_STEP_SIZE / 2.0f, VECTOR_BUFF[1]);
	multLOnVector(curState, k2);

#ifdef DEBUG_CONTINUOUS
	LOG_IF_APPROPRIATE(print(consoleStream, "k2", k2, basisSize));
#endif

	//same but with another temporary vector for the buffer
	complex_cblas_copy(basisSize, prevState, NO_INC, curState, NO_INC);
	complex_cblas_axpy(basisSize, &complexTHalfStep, k2, NO_INC, curState,
			NO_INC);
	//k3 = f(t + T_STEP_SIZE / 2.0f, VECTOR_BUFF[2])
	multLOnVector(curState, k3);

#ifdef DEBUG_CONTINUOUS
	LOG_IF_APPROPRIATE(print(consoleStream, "k3", k3, basisSize));
#endif

	complex_cblas_copy(basisSize, prevState, NO_INC, curState, NO_INC);
	complex_cblas_axpy(basisSize, &complexTStep, k3, NO_INC, curState, NO_INC);
	//k4 = f(t + T_STEP_SIZE, VECTOR_BUFF[3]);
	multLOnVector(curState, k4);

#ifdef DEBUG_CONTINUOUS
	LOG_IF_APPROPRIATE(print(consoleStream, "k4", k4, basisSize));
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

inline void Solver::multLOnVector(COMPLEX_TYPE *vector, COMPLEX_TYPE *result) {
#ifdef H_SPARSE
	complex_mkl_cspblas_csrgemv("n", &basisSize, lCSR3->values, lCSR3->rowIndex,
			lCSR3->columns, vector, result);
#else
	cublasStatus_t cublasStatus = cublasSetVector(basisSize,
			sizeof(COMPLEX_TYPE), vector, NO_INC, devPtrVector, NO_INC);
	if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
		std::cout
				<< std::string(
						"Transfer of the state vector to the device memory failed with error:")
						+ std::to_string(cublasStatus) << endl;
		throw cublasStatus;
	}

	cublasStatus =
			cublasgemv(cublasHandle, CUBLAS_OP_N, basisSize, basisSize,
					reinterpret_cast<CUDA_COMPLEX_TYPE *>(const_cast<COMPLEX_TYPE *>(&complexOne)),
					devPtrL, basisSize, devPtrVector, NO_INC,
					reinterpret_cast<CUDA_COMPLEX_TYPE *>(const_cast<COMPLEX_TYPE *>(&complexZero)),
					devPtrResult, NO_INC);
	if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
		std::cout
				<< std::string("GEMV failed with error:")
						+ std::to_string(cublasStatus) << endl;
		throw cublasStatus;
	}

	cublasStatus = cublasGetVector(basisSize, sizeof(CUDA_COMPLEX_TYPE),
			devPtrResult, NO_INC, result, NO_INC);
	if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
		std::cout
				<< std::string("Can't retrieve result from device:")
						+ std::to_string(cublasStatus) << endl;
		throw cublasStatus;
	}
#endif
}

inline void Solver::normalizeVector(COMPLEX_TYPE *stateVector) {
//calculate new norm
	complex_cblas_dotc_sub(basisSize, stateVector, NO_INC, stateVector, NO_INC,
			&norm2);

	normalizeVector(stateVector, norm2, tempVector);

//write back
	complex_cblas_copy(basisSize, tempVector, NO_INC, stateVector, NO_INC);
}

inline void Solver::normalizeVector(COMPLEX_TYPE *stateVector,
		const COMPLEX_TYPE &stateVectorNorm2,
		COMPLEX_TYPE *result) {

	vSqrt((MKL_INT) 1, &(stateVectorNorm2.real), &(normReversed.real));
	normReversed.real = 1.0 / normReversed.real;

	complex_cblas_copy(basisSize, zeroVector, NO_INC, result, NO_INC);
	complex_cblas_axpy(basisSize, &normReversed, stateVector, NO_INC, result,
			NO_INC);
}

inline void Solver::makeJump(std::ostream &consoleStream,
COMPLEX_TYPE *prevState, COMPLEX_TYPE *curState) {
//then a jump is occurred between t(i) and t(i+1)
//let's suppose it was at time t(i)

#ifdef DEBUG_JUMPS
	print(consoleStream, "State at the jump moment", prevState, basisSize);
#endif

//calculate vectors and their norms after each type of jump
//the jump is made from the previous step state
//(in the first cavity, in the second or in the third)
	complex_mkl_cspblas_csrgemv("n", &basisSize, a1CSR3->values,
			a1CSR3->rowIndex, a1CSR3->columns, prevState, k1);
	complex_cblas_dotc_sub(basisSize, k1, NO_INC, k1, NO_INC, &n12);

	complex_mkl_cspblas_csrgemv("n", &basisSize, a2CSR3->values,
			a2CSR3->rowIndex, a2CSR3->columns, prevState, k2);
	complex_cblas_dotc_sub(basisSize, k2, NO_INC, k2, NO_INC, &n22);

	complex_mkl_cspblas_csrgemv("n", &basisSize, a3CSR3->values,
			a3CSR3->rowIndex, a3CSR3->columns, prevState, k3);
	complex_cblas_dotc_sub(basisSize, k3, NO_INC, k3, NO_INC, &n32);

//calculate probabilities of each jump
	FLOAT_TYPE n2Sum = n12.real + n22.real + n32.real;
	FLOAT_TYPE p1 = n12.real / n2Sum;
	FLOAT_TYPE p12 =
			p1
					+ n22.real / n2Sum;	//two first cavities together

#ifdef DEBUG_JUMPS
					print(consoleStream, "If jump will be in the first cavity", k1, basisSize);
					consoleStream << "it's norm^2: " << n12.real << endl;

					print(consoleStream, "If jump will be in the second cavity", k2, basisSize);
					consoleStream << "it's norm^2: " << n22.real << endl;

					print(consoleStream, "If jump will be in the third cavity", k3, basisSize);
					consoleStream << "it's norm^2: " << n32.real << endl;

					consoleStream << "Probabilities of the jumps: in the first = " << p1
					<< ", (the first + the second) = " << p12 << ", third = " << 1-p12 << endl;
#endif

//choose which jump is occurred,
	FLOAT_TYPE rnd = nextRandom();
	if (rnd < p1) {
#ifdef DEBUG_JUMPS
		consoleStream << "It jumped in the FIRST cavity" << endl;
#endif
		normalizeVector(k1, n12, curState);
	} else if (rnd < p12) {
#ifdef DEBUG_JUMPS
		consoleStream << "Jumped in the SECOND cavity" << endl;
#endif
		normalizeVector(k2, n22, curState);
	} else {
#ifdef DEBUG_JUMPS
		consoleStream << "Jumped in the THIRD cavity" << endl;
#endif
		normalizeVector(k3, n32, curState);
	}

#ifdef DEBUG_JUMPS
	print(consoleStream, "State vector after the jump and normalization",
			curState, basisSize);
#endif
}

inline FLOAT_TYPE Solver::nextRandom() {
//check whether there are random numbers left
	if (rndNumIndex == RND_NUM_BUFF_SIZE) {
		rndNumProvider.initBuffer(id, rndNumBuff, RND_NUM_BUFF_SIZE);
		rndNumIndex = 0;
	}

	return rndNumBuff[rndNumIndex++];
}

