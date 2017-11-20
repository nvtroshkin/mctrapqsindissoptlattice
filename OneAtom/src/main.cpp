#include <iostream>
#include <mkl.h>
#include <mkl_types.h>
#include <mkl_vsl.h>
#include <scalable_allocator.h>
#include <math.h>
#include <chrono>

#include <eval-params.h>
#include <system-constants.h>
#include <runge-kutta.h>

using namespace std;

// Use typed constants instead of #define
static const int RND_NUM_BUFF_SIZE = 8 * 1024;
static const int SIMDALIGN = 1024; //an alignment for memory in a multithreaded environment
//a constant - initializer of a pseudorandom numbers generator
#define RANDSEED 777

static const float T_STEP_SIZE = EVAL_TIME / TIME_STEPS_NUMBER;
static const MKL_INT NO_INC = 1; //for the BLAS vector library - not incrementing vectors

//A sample is a two-dimensional array of slices of the state vector at the steps of discretized time
//TIME_STEPS_NUMBER + 1 because of elements of the array are points, not segments
static MKL_Complex8 samples[MONTE_CARLO_SAMPLES_NUMBER][TIME_STEPS_NUMBER + 1][DRESSED_BASIS_SIZE];

static MKL_Complex8 VECTOR_BUFF[7][DRESSED_BASIS_SIZE]; //a storage for temporary vectors
static const MKL_Complex8 MULT_T_HALF_STEP = MKL_Complex8 { T_STEP_SIZE / 2.0f,
		0.0f };
static const MKL_Complex8 MULT_T_STEP = MKL_Complex8 { T_STEP_SIZE, 0.0f };
static const MKL_Complex8 MULT_T_SIXTH_STEP = MKL_Complex8 { T_STEP_SIZE / 6.0f,
		0.0f };
static const MKL_Complex8 MULT_TWO = MKL_Complex8 { 2.0f, 0.0f };
static const MKL_Complex8 MULT_ONE = MKL_Complex8 { 1.0f, 0.0f };

int main(int argc, char **argv) {
	auto start = chrono::steady_clock::now();

	//init cache
	initPhotonNumbersSqrts();

	//The basis vectors are enumerated flatly, |photon number>|atom state>, |0>|0>, |0>|1> and etc.

	MatrixDiagForm diagH = getHhatInDiagForm();

	//a matrix in the diagonal storage format
	const int ADistanceLength = 1;
	const int ADistance[1] = { -2 };
	MKL_Complex8 ADiagValues[DRESSED_BASIS_SIZE];

	for (int i = 0; i < DRESSED_BASIS_SIZE; i++) {
		ADiagValues[i] = {aPlus(i, i-2),0.0f};
	}

	MKL_Complex8 zeroVector[DRESSED_BASIS_SIZE];
	for (int i = 0; i < DRESSED_BASIS_SIZE; i++) {
		zeroVector[i] = {0.0f,0.0f};
	}

	//Initialize each sample by the ground state vector
	for (int i = 0; i < MONTE_CARLO_SAMPLES_NUMBER; i++) {
		cblas_ccopy((MKL_INT) DRESSED_BASIS_SIZE, zeroVector, NO_INC,
				samples[i][0], NO_INC);

		samples[i][0][0] = {1.0f,0.0f};
	}

	//create a random numbers stream
	int rndNumIndex = 0;		//indicates where we are in the buffer
	float* rndNumBuff = (float *) scalable_aligned_malloc(
			RND_NUM_BUFF_SIZE * sizeof(float), SIMDALIGN);
	VSLStreamStatePtr Stream;
	vslNewStream(&Stream, VSL_BRNG_MCG31, RANDSEED);
	vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, Stream, RND_NUM_BUFF_SIZE,
			rndNumBuff, 0.0f, 1.0f);

	const int *HDistanceLength = &diagH.diagDistLength;
	const int *HDistance = diagH.diagsDistances;
	const MKL_Complex8 *HDiagValues = diagH.matrix;

	MKL_Complex8 k1[DRESSED_BASIS_SIZE];
	MKL_Complex8 k2[DRESSED_BASIS_SIZE];
	MKL_Complex8 k3[DRESSED_BASIS_SIZE];
	MKL_Complex8 k4[DRESSED_BASIS_SIZE];
	MKL_Complex8 norm = { 1.0f, 0.0f };
	for (int sampleIndex = 0; sampleIndex < MONTE_CARLO_SAMPLES_NUMBER;
			sampleIndex++) {
//		float t = 0;	//each sample starts at t=0
		MKL_Complex8 (*sample)[DRESSED_BASIS_SIZE] = samples[sampleIndex];

		//Calculate each sample by the time axis
		for (int i = 0; i < TIME_STEPS_NUMBER; i++) {
			//Before the next jump there is deterministic evolution guided by
			//the Shroedinger's equation

			//A jump occurs between t(i) and t(i+1) when square of the state vector norm
			//at t(i+1)...
			for (int j = 0; j < DRESSED_BASIS_SIZE; j++) {
				//write the f function

				//k1 = f(t, sample[i]);
				//to k1
				mkl_cdiasymv("l", &(DRESSED_BASIS_SIZE), HDiagValues,
						&(DRESSED_BASIS_SIZE), HDistance, HDistanceLength,
						sample[i], k1);

				//copy current state to a temporary vector
				cblas_ccopy((MKL_INT) DRESSED_BASIS_SIZE, sample[i], NO_INC,
						VECTOR_BUFF[1], NO_INC);
				cblas_caxpy((MKL_INT) DRESSED_BASIS_SIZE, &MULT_T_HALF_STEP, k1,
						NO_INC, VECTOR_BUFF[1], NO_INC);
				//k2 = f(t + T_STEP_SIZE / 2.0f, VECTOR_BUFF[1]);
				mkl_cdiasymv("l", &(DRESSED_BASIS_SIZE), HDiagValues,
						&(DRESSED_BASIS_SIZE), HDistance, HDistanceLength,
						VECTOR_BUFF[1], k2);

				//same but with another temporary vector for the buffer
				cblas_ccopy((MKL_INT) DRESSED_BASIS_SIZE, sample[i], NO_INC,
						VECTOR_BUFF[2], NO_INC);
				cblas_caxpy((MKL_INT) DRESSED_BASIS_SIZE, &MULT_T_HALF_STEP, k2,
						NO_INC, VECTOR_BUFF[2], NO_INC);
				//k3 = f(t + T_STEP_SIZE / 2.0f, VECTOR_BUFF[2])
				mkl_cdiasymv("l", &(DRESSED_BASIS_SIZE), HDiagValues,
						&(DRESSED_BASIS_SIZE), HDistance, HDistanceLength,
						VECTOR_BUFF[2], k3);

				cblas_ccopy((MKL_INT) DRESSED_BASIS_SIZE, sample[i], NO_INC,
						VECTOR_BUFF[3], NO_INC);
				cblas_caxpy((MKL_INT) DRESSED_BASIS_SIZE, &MULT_T_STEP, k3,
						NO_INC, VECTOR_BUFF[3], NO_INC);
				//k4 = f(t + T_STEP_SIZE, VECTOR_BUFF[3]);
				mkl_cdiasymv("l", &(DRESSED_BASIS_SIZE), HDiagValues,
						&(DRESSED_BASIS_SIZE), HDistance, HDistanceLength,
						VECTOR_BUFF[3], k4);

				//store to k1
				cblas_caxpy((MKL_INT) DRESSED_BASIS_SIZE, &MULT_TWO, k2, NO_INC,
						k1, NO_INC);
				//to k4
				cblas_caxpy((MKL_INT) DRESSED_BASIS_SIZE, &MULT_TWO, k3, NO_INC,
						k4, NO_INC);
				//to k1
				cblas_caxpy((MKL_INT) DRESSED_BASIS_SIZE, &MULT_ONE, k4, NO_INC,
						k1, NO_INC);
				//modify sample[i+1]
				cblas_ccopy((MKL_INT) DRESSED_BASIS_SIZE, sample[i], NO_INC,
						sample[i + 1], NO_INC);
				cblas_caxpy((MKL_INT) DRESSED_BASIS_SIZE, &MULT_T_SIXTH_STEP,
						k1, NO_INC, sample[i + 1], NO_INC);
			}

			//...falls below the threshold, which is a random number
			//uniformly distributed between [0,1]
			float svNormThreshold = rndNumBuff[rndNumIndex++];

			//if the state vector at t(i+1) has a less square of the norm then the threshold
			//try a self-written norm?
			cblas_cdotc_sub((MKL_INT) DRESSED_BASIS_SIZE, sample[i + 1], NO_INC,
					sample[i + 1], NO_INC, &norm);
			if (svNormThreshold > norm.real) {
				//then a jump is occurred between t(i) and t(i+1)
				//let's suppose it was at time t(i)

				//calculate the state vector after the jump
				//store it at t(i+1)
				mkl_cdiagemv("n", &(DRESSED_BASIS_SIZE), ADiagValues,
						&(DRESSED_BASIS_SIZE), ADistance, &ADistanceLength,
						sample[i], sample[i + 1]);
				//calculate new norm
				cblas_cdotc_sub((MKL_INT) DRESSED_BASIS_SIZE, sample[i + 1],
						NO_INC, sample[i + 1], NO_INC, &norm);
			}

			//normalize vector
			cblas_ccopy((MKL_INT) DRESSED_BASIS_SIZE, zeroVector, NO_INC,
					VECTOR_BUFF[0], NO_INC);
			cblas_caxpy((MKL_INT) DRESSED_BASIS_SIZE, &norm, sample[i + 1],
					NO_INC, VECTOR_BUFF[0], NO_INC);
			//write back
			cblas_ccopy((MKL_INT) DRESSED_BASIS_SIZE, VECTOR_BUFF[0], NO_INC,
					sample[i + 1], NO_INC);
		}
	}

	//now the Monte-Carlo simulations are over
	//we should calculate the mean photons number and the variance

	//an auxiliary array with photon numbers for each basis vector is needed
	MKL_Complex8 statePhotonNumber[DRESSED_BASIS_SIZE];
	for (int i = 0; i < DRESSED_BASIS_SIZE; i++) {
		statePhotonNumber[i] = {(float)n(i),0.0f};
	}

	//Sum(<psi|n|psi>)
	//mult psi on ns
	float meanPhotonsNumber = 0.0f;
	float meanPhotonNumbers[MONTE_CARLO_SAMPLES_NUMBER];
	for (int i = 0; i < MONTE_CARLO_SAMPLES_NUMBER; i++) {
		vcMul((MKL_INT) DRESSED_BASIS_SIZE, samples[i][TIME_STEPS_NUMBER],
				statePhotonNumber, VECTOR_BUFF[0]);
		cblas_cdotc_sub((MKL_INT) DRESSED_BASIS_SIZE, VECTOR_BUFF[0], NO_INC,
				samples[i][TIME_STEPS_NUMBER], NO_INC, &norm);

		meanPhotonsNumber += norm.real;
		//store for the variance
		meanPhotonNumbers[i] = norm.real;
	}
	meanPhotonsNumber /= MONTE_CARLO_SAMPLES_NUMBER;

	//variance
	float sum = 0.0;
	for (int i = 0; i < MONTE_CARLO_SAMPLES_NUMBER; i++) {
		sum += (meanPhotonNumbers[i] - meanPhotonsNumber)
				* (meanPhotonNumbers[i] - meanPhotonsNumber);
	}

	float variance = sqrtf(
			sum / MONTE_CARLO_SAMPLES_NUMBER
					/ (MONTE_CARLO_SAMPLES_NUMBER - 1));

	cout << "Mean photons number: " << meanPhotonsNumber << "\n";
	cout << "Variance: " << variance << endl;

	auto end = chrono::steady_clock::now();
	auto diff = end - start;
	cout << "Elapsed time is :  " << chrono::duration_cast < chrono::nanoseconds
			> (diff).count() / 1000000000.0 << "s" << endl;

	return 0;
}
