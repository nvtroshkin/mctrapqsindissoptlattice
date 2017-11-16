#include <iostream>
#include <mkl.h>
#include <mkl_types.h>
#include <mkl_vsl.h>
#include <scalable_allocator.h>

using namespace std;

#define MONTE_CARLO_SAMPLES_NUMBER 10
//the dressed basis macros
static const int MAX_PHOTON_NUMBER = 2;

//Evaluation of each sample is performed beginning at 0s and ending at the end time.
//Increasing the END_TIME value is necessary to caught the stationary evaluation
//phase
#define EVAL_TIME 10
#define TIME_STEPS_NUMBER 100		//the total number of steps by the time axis

// Use typed constants instead of #define
static const int RND_NUM_BUFF_SIZE = 8 * 1024;
static const int SIMDALIGN = 1024; //an alignment for memory in a multithreaded environment
//a constant - initializer of a pseudorandom numbers generator
#define RANDSEED 777

static const int DRESSED_BASIS_SIZE = (2 * (MAX_PHOTON_NUMBER + 1));
static const float T_STEP_SIZE = 1.0f * EVAL_TIME / TIME_STEPS_NUMBER;
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

static float sqrtsOfPhotonNumbers[MAX_PHOTON_NUMBER + 1];

static const float KAPPA = 1.0f;
static const float DELTA_OMEGA = 20.0f;
static const float G = 50.0f;
static const float LATIN_E = 2.0f;

int n(int index) {
	return index / 2;
}

int s(int index) {
	return index % 2;
}

//to obtain a just swap i and j
float aPlus(int i, int j) {
	if (n(i) != n(j) + 1 || s(i) != s(j)) {
		return 0.0f;
	}

	return sqrtsOfPhotonNumbers[n(j) + 1];
}

//to obtain sigmaMinus just swap i and j
float sigmaPlus(int i, int j) {
	if (s(j) != 0 || s(i) != 1 || n(i) != n(j)) {
		return 0.0f;
	}

	return 1.0f;
}

//hbar = 1
//The Hhat from Petruchionne p363, the (7.11) expression
MKL_Complex8 H(int i, int j) {
	//the real part of the matrix element
	float imaginary = 0.0f;
	for (int k = 0; k < DRESSED_BASIS_SIZE; k++) {
		imaginary -=
				-DELTA_OMEGA
						* (aPlus(i, k) * aPlus(k, i)
								+ sigmaPlus(i, k) * sigmaPlus(k, i))
						+ G
								* (aPlus(k, i) * sigmaPlus(k, j)
										+ aPlus(i, k) * sigmaPlus(j, k));
	}

	imaginary -= LATIN_E * (aPlus(i, j) + aPlus(j, i));

	//the imaginary
	float real = 0.0;
	for (int k = 0; k < DRESSED_BASIS_SIZE; k++) {
		real -= aPlus(i, k) * aPlus(j, k);
	}

	real *= KAPPA;

	return {real, imaginary};
}

int main(int argc, char **argv) {
	//The basis vectors are enumerated flatly, |photon number>|atom state>, |0>|0>, |0>|1> and etc.

	//Hhat is a 5-diagonal symmetrical matrix
	//it can be stored by calculating the main and two lower diagonals
	//and stored into the diagonal matrix storage format (https://software.intel.com/en-us/mkl-developer-reference-c-sparse-blas-diagonal-matrix-storage-format)
	//calculate square roots
	float photonNumbers[MAX_PHOTON_NUMBER + 1];
	for (int k = 0; k < MAX_PHOTON_NUMBER; k++) {
		photonNumbers[k] = k;
	}

	vsSqrt((MKL_INT) (MAX_PHOTON_NUMBER + 1), photonNumbers,
			sqrtsOfPhotonNumbers);

	//diagonals in relation to the main diagonal
	//only 5
	const int distanceLength = 5;
	const int distance[5] = { -2, -1, 0, 1, 2 };

	//the compact diagonals storage
	MKL_Complex8 HDiagValues[DRESSED_BASIS_SIZE * 5];

	//the diagonals have different lengths. It is convenient to make them
	//the same length but then some indices violate the array boundaries. We should
	//cut them in the inner cycle making as less conditions checks as we can
	float modifiedLength;
	for (int i = -2; i <= 2; i++) {
		modifiedLength = DRESSED_BASIS_SIZE + (i > 0 ? 0 : i);
		for (int j = (i > 0 ? i : 0); j < modifiedLength; j++) {
			HDiagValues[j * distanceLength + i + 2] = H(j - i, j);
		}
	}

	//Initial samples state is the ground state
	for (int i = 0; i < MONTE_CARLO_SAMPLES_NUMBER; i++) {
		for (int j = 0; j < DRESSED_BASIS_SIZE; j++) {
			samples[i][0][j] = {0.0f,0.0f};
		}

		//the ground state initialization
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

	MKL_Complex8 k1[DRESSED_BASIS_SIZE];
	MKL_Complex8 k2[DRESSED_BASIS_SIZE];
	MKL_Complex8 k3[DRESSED_BASIS_SIZE];
	MKL_Complex8 k4[DRESSED_BASIS_SIZE];
	for (int sampleIndex = 0; sampleIndex < MONTE_CARLO_SAMPLES_NUMBER;
			sampleIndex++) {
		float t = 0;	//each sample starts at t=0
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
						&(DRESSED_BASIS_SIZE), distance, &distanceLength,
						sample[i], k1);

				//copy current state to a temporary vector
				cblas_ccopy((MKL_INT) DRESSED_BASIS_SIZE, sample[i], NO_INC,
						VECTOR_BUFF[1], NO_INC);
				cblas_caxpy((MKL_INT) DRESSED_BASIS_SIZE, &MULT_T_HALF_STEP, k1,
						NO_INC, VECTOR_BUFF[1], NO_INC);
				//k2 = f(t + T_STEP_SIZE / 2.0f, VECTOR_BUFF[1]);
				mkl_cdiasymv("l", &(DRESSED_BASIS_SIZE), HDiagValues,
						&(DRESSED_BASIS_SIZE), distance, &distanceLength,
						VECTOR_BUFF[1], k2);

				//same but with another temporary vector for the buffer
				cblas_ccopy((MKL_INT) DRESSED_BASIS_SIZE, sample[i], NO_INC,
						VECTOR_BUFF[2], NO_INC);
				cblas_caxpy((MKL_INT) DRESSED_BASIS_SIZE, &MULT_T_HALF_STEP, k2,
						NO_INC, VECTOR_BUFF[2], NO_INC);
				//k3 = f(t + T_STEP_SIZE / 2.0f, VECTOR_BUFF[2])
				mkl_cdiasymv("l", &(DRESSED_BASIS_SIZE), HDiagValues,
						&(DRESSED_BASIS_SIZE), distance, &distanceLength,
						VECTOR_BUFF[2], k3);

				cblas_ccopy((MKL_INT) DRESSED_BASIS_SIZE, sample[i], NO_INC,
						VECTOR_BUFF[3], NO_INC);
				cblas_caxpy((MKL_INT) DRESSED_BASIS_SIZE, &MULT_T_STEP, k3,
						NO_INC, VECTOR_BUFF[3], NO_INC);
				//k4 = f(t + T_STEP_SIZE, VECTOR_BUFF[3]);
				mkl_cdiasymv("l", &(DRESSED_BASIS_SIZE), HDiagValues,
						&(DRESSED_BASIS_SIZE), distance, &distanceLength,
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
			if (sample[i])

				//then a jump is occurred between t(i) and t(i+1)
				//calculate the state vector after the jump
				//store it at t(i+1)
				//else
				//make a deterministic evolution step
				if () {

				}
		}
	}

	return 0;
}
