#include <iostream>
#include <fstream>
#include <mkl.h>
#include <mkl_types.h>
#include <mkl_vsl.h>
#include <scalable_allocator.h>
#include <math.h>
#include <cmath>
#include <chrono>
#include <omp.h>

#include <macro.h>
#include <eval-params.h>
#include <system-constants.h>
#include <runge-kutta.h>

using namespace std;

static const MKL_INT NO_INC = 1; //for the BLAS vector library - not incrementing vectors

static COMPLEX_TYPE VECTOR_BUFF[7][DRESSED_BASIS_SIZE]; //a storage for temporary vectors
static const COMPLEX_TYPE MULT_T_HALF_STEP = COMPLEX_TYPE { 0.5f * T_STEP_SIZE,
		0.0f };
static const COMPLEX_TYPE MULT_T_STEP = COMPLEX_TYPE { T_STEP_SIZE, 0.0f };
static const COMPLEX_TYPE MULT_T_SIXTH_STEP = COMPLEX_TYPE { T_STEP_SIZE / 6.0f,
		0.0f };
static const COMPLEX_TYPE MULT_TWO = COMPLEX_TYPE { 2.0f, 0.0f };
static const COMPLEX_TYPE MULT_ONE = COMPLEX_TYPE { 1.0f, 0.0f };

//A sample is a two-dimensional array of slices of the state vector at the steps of discretized time
//TIME_STEPS_NUMBER + 1 because of elements of the array are points, not segments
static COMPLEX_TYPE samples[MONTE_CARLO_SAMPLES_NUMBER][TIME_STEPS_NUMBER + 1][DRESSED_BASIS_SIZE];

inline void normalizeVector(COMPLEX_TYPE normReversed, COMPLEX_TYPE norm2,
COMPLEX_TYPE *zeroVector, COMPLEX_TYPE *stateVector) {

	vsSqrt((MKL_INT) 1, &norm2.real, &normReversed.real);
	normReversed.real = 1.0f / normReversed.real;

	cblas_ccopy((MKL_INT) DRESSED_BASIS_SIZE, zeroVector, NO_INC,
			VECTOR_BUFF[0], NO_INC);
	cblas_caxpy((MKL_INT) DRESSED_BASIS_SIZE, &normReversed, stateVector,
			NO_INC, VECTOR_BUFF[0], NO_INC);
	//write back
	cblas_ccopy((MKL_INT) DRESSED_BASIS_SIZE, VECTOR_BUFF[0], NO_INC,
			stateVector, NO_INC);
}

inline void make4thOrderRungeKuttaStep(const COMPLEX_TYPE *HCSR3Values,
		const int *HCSR3RowIndex, const int *HCSR3Columns,
		COMPLEX_TYPE *curState, COMPLEX_TYPE *nextState,
		//auxiliary variables
		COMPLEX_TYPE *k1, COMPLEX_TYPE *k2, COMPLEX_TYPE *k3,
		COMPLEX_TYPE *k4) {
	//k1 = f(t, sample[i]);
	//to k1
	mkl_cspblas_ccsrgemv("n", &(DRESSED_BASIS_SIZE), HCSR3Values, HCSR3RowIndex,
			HCSR3Columns, curState, k1);

	//copy current state to a temporary vector
	cblas_ccopy((MKL_INT) DRESSED_BASIS_SIZE, curState, NO_INC, VECTOR_BUFF[1],
			NO_INC);
	cblas_caxpy((MKL_INT) DRESSED_BASIS_SIZE, &MULT_T_HALF_STEP, k1, NO_INC,
			VECTOR_BUFF[1], NO_INC);
	//k2 = f(t + T_STEP_SIZE / 2.0f, VECTOR_BUFF[1]);
	mkl_cspblas_ccsrgemv("n", &(DRESSED_BASIS_SIZE), HCSR3Values, HCSR3RowIndex,
			HCSR3Columns, VECTOR_BUFF[1], k2);

	//same but with another temporary vector for the buffer
	cblas_ccopy((MKL_INT) DRESSED_BASIS_SIZE, curState, NO_INC, VECTOR_BUFF[2],
			NO_INC);
	cblas_caxpy((MKL_INT) DRESSED_BASIS_SIZE, &MULT_T_HALF_STEP, k2, NO_INC,
			VECTOR_BUFF[2], NO_INC);
	//k3 = f(t + T_STEP_SIZE / 2.0f, VECTOR_BUFF[2])
	mkl_cspblas_ccsrgemv("n", &(DRESSED_BASIS_SIZE), HCSR3Values, HCSR3RowIndex,
			HCSR3Columns, VECTOR_BUFF[2], k3);

	cblas_ccopy((MKL_INT) DRESSED_BASIS_SIZE, curState, NO_INC, VECTOR_BUFF[3],
			NO_INC);
	cblas_caxpy((MKL_INT) DRESSED_BASIS_SIZE, &MULT_T_STEP, k3, NO_INC,
			VECTOR_BUFF[3], NO_INC);
	//k4 = f(t + T_STEP_SIZE, VECTOR_BUFF[3]);
	mkl_cspblas_ccsrgemv("n", &(DRESSED_BASIS_SIZE), HCSR3Values, HCSR3RowIndex,
			HCSR3Columns, VECTOR_BUFF[3], k4);

	//store to k1
	cblas_caxpy((MKL_INT) DRESSED_BASIS_SIZE, &MULT_TWO, k2, NO_INC, k1,
			NO_INC);
	//to k4
	cblas_caxpy((MKL_INT) DRESSED_BASIS_SIZE, &MULT_TWO, k3, NO_INC, k4,
			NO_INC);
	//to k1
	cblas_caxpy((MKL_INT) DRESSED_BASIS_SIZE, &MULT_ONE, k4, NO_INC, k1,
			NO_INC);
	//modify sample[i+1]
	cblas_ccopy((MKL_INT) DRESSED_BASIS_SIZE, curState, NO_INC, nextState,
			NO_INC);
	cblas_caxpy((MKL_INT) DRESSED_BASIS_SIZE, &MULT_T_SIXTH_STEP, k1, NO_INC,
			nextState, NO_INC);
}

void print(char title[], COMPLEX_TYPE array[]) {
	cout << title << ": {" << endl;
	for (int v = 0; v < DRESSED_BASIS_SIZE; v++) {
		cout << array[v].real << " + " << array[v].imag << "i, ";
	}
	cout << "}" << endl;
}

int main(int argc, char **argv) {
	auto start = chrono::steady_clock::now();

	//The basis vectors are enumerated flatly, |photon number>|atom state>, |0>|0>, |0>|1> and etc.

	//init cache
	initPhotonNumbersSqrts();

	COMPLEX_TYPE zeroVector[DRESSED_BASIS_SIZE];
	for (int i = 0; i < DRESSED_BASIS_SIZE; i++) {
		zeroVector[i] = {0.0f,0.0f};
	}

	//create a random numbers stream
	int rndNumIndex = 0;		//indicates where we are in the buffer
	FLOAT_TYPE* rndNumBuff = (FLOAT_TYPE *) scalable_aligned_malloc(
			RND_NUM_BUFF_SIZE * sizeof(FLOAT_TYPE), SIMDALIGN);
//	for (int i = 0; i < RND_NUM_BUFF_SIZE; i++) {
//		rndNumBuff[i]=0.0;
//	}
	VSLStreamStatePtr Stream;
	vslNewStream(&Stream, VSL_BRNG_MCG31, RANDSEED);
	vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, Stream, RND_NUM_BUFF_SIZE,
			rndNumBuff, 0.0f, 1.0f);
//	ofstream myfile;
//	myfile.open("rnd-numbers.txt");
//	if (!myfile.is_open()) {
//		cout << "Can't open file!" << endl;
//	}
//	for (int i = 0; i < RND_NUM_BUFF_SIZE; i++) {
//		myfile << rndNumBuff[i] << ", ";
//	}
//	myfile.close();

	CSR3Matrix hCSR3 = getHInCSR3();
	const int *HCSR3RowIndex = hCSR3.rowIndex;
	const int *HCSR3Columns = hCSR3.columns;
	const COMPLEX_TYPE *HCSR3Values = hCSR3.values;

	CSR3Matrix aCSR3 = getAInCSR3();
	const int *aCSR3RowIndex = aCSR3.rowIndex;
	const int *aCSR3Columns = aCSR3.columns;
	const COMPLEX_TYPE *aCSR3Values = aCSR3.values;

	//Initialize each sample by the ground state vector
	for (int i = 0; i < MONTE_CARLO_SAMPLES_NUMBER; i++) {
		cblas_ccopy((MKL_INT) DRESSED_BASIS_SIZE, zeroVector, NO_INC,
				samples[i][0], NO_INC);

		samples[i][0][0] = {1.0f,0.0f};
	}

	//initialize the threshold for the random jump identification
	FLOAT_TYPE svNormThreshold;
	COMPLEX_TYPE (*sample)[DRESSED_BASIS_SIZE];

	//may be non-sparse calculation would be faster?
	COMPLEX_TYPE k1[DRESSED_BASIS_SIZE];
	COMPLEX_TYPE k2[DRESSED_BASIS_SIZE];
	COMPLEX_TYPE k3[DRESSED_BASIS_SIZE];
	COMPLEX_TYPE k4[DRESSED_BASIS_SIZE];
	COMPLEX_TYPE norm2 = { 1.0f, 0.0f };
	COMPLEX_TYPE normReversed = { 1.0f, 0.0f };
//#pragma omp parallel for
	for (int sampleIndex = 0; sampleIndex < MONTE_CARLO_SAMPLES_NUMBER;
			sampleIndex++) {
//		FLOAT_TYPE t = 0;	//each sample starts at t=0
		sample = samples[sampleIndex];
		svNormThreshold = rndNumBuff[rndNumIndex++];

		//Calculate each sample by the time axis
		for (int i = 0; i < TIME_STEPS_NUMBER; i++) {
			//Before the next jump there is deterministic evolution guided by
			//the Shroedinger's equation

			//A jump occurs between t(i) and t(i+1) when square of the state vector norm
			//at t(i+1)...
			//write the f function

			make4thOrderRungeKuttaStep(HCSR3Values, HCSR3RowIndex, HCSR3Columns,
					sample[i], sample[i + 1], k1, k2, k3, k4);

			//...falls below the threshold, which is a random number
			//uniformly distributed between [0,1] - svNormThreshold

			//if the state vector at t(i+1) has a less square of the norm then the threshold
			//try a self-written norm?
			cblas_cdotc_sub((MKL_INT) DRESSED_BASIS_SIZE, sample[i + 1], NO_INC,
					sample[i + 1], NO_INC, &norm2);
			if (svNormThreshold > norm2.real) {
				//then a jump is occurred between t(i) and t(i+1)
				//let's suppose it was at time t(i)

				//calculate the state vector after the jump
				//store it at t(i+1)
				mkl_cspblas_ccsrgemv("n", &(DRESSED_BASIS_SIZE), aCSR3Values,
						aCSR3RowIndex, aCSR3Columns, sample[i], sample[i + 1]);
				//calculate new norm
				cblas_cdotc_sub((MKL_INT) DRESSED_BASIS_SIZE, sample[i + 1],
						NO_INC, sample[i + 1], NO_INC, &norm2);

				//update the random time
				svNormThreshold = rndNumBuff[rndNumIndex++];

				//normalize vector
				normalizeVector(normReversed, norm2, zeroVector, sample[i + 1]);
			}
		}

		//final state normalization
		normalizeVector(normReversed, norm2, zeroVector,
				sample[TIME_STEPS_NUMBER]);
	}

	//check if random numbers valid - replace with correct random numbers generation
	if (rndNumIndex >= RND_NUM_BUFF_SIZE) {
		cout << "No more random numbers" << endl;
	}

	//now the Monte-Carlo simulations are over
	//we should calculate the mean photons number and the variance

	//an auxiliary array with photon numbers for each basis vector is needed
	COMPLEX_TYPE statePhotonNumber[DRESSED_BASIS_SIZE];
	for (int i = 0; i < DRESSED_BASIS_SIZE; i++) {
		statePhotonNumber[i] = {(FLOAT_TYPE)n(i),0.0f};
	}

	//Sum(<psi|n|psi>)
	//mult psi on ns
	FLOAT_TYPE meanPhotonNumbers[MONTE_CARLO_SAMPLES_NUMBER];
	for (int i = 0; i < MONTE_CARLO_SAMPLES_NUMBER; i++) {
		vcMul((MKL_INT) DRESSED_BASIS_SIZE, samples[i][TIME_STEPS_NUMBER],
				statePhotonNumber, VECTOR_BUFF[0]);
		cblas_cdotc_sub((MKL_INT) DRESSED_BASIS_SIZE, VECTOR_BUFF[0], NO_INC,
				samples[i][TIME_STEPS_NUMBER], NO_INC, &norm2);

				//store for the variance
		meanPhotonNumbers[i] = norm2.real;
	}

	FLOAT_TYPE meanPhotonsNumber = cblas_sasum((MKL_INT)MONTE_CARLO_SAMPLES_NUMBER, meanPhotonNumbers, NO_INC);
	meanPhotonsNumber /= MONTE_CARLO_SAMPLES_NUMBER;

	//variance. Calculate like this to avoid close numbers subtraction
	//Sum(mean photon numbers)^2
	FLOAT_TYPE sum1 = cblas_sdsdot((MKL_INT) MONTE_CARLO_SAMPLES_NUMBER, 0.0f, meanPhotonNumbers, NO_INC,
					meanPhotonNumbers, NO_INC);

	//Sum(2*mean photon number*mean photon number[i])
	FLOAT_TYPE temp[DRESSED_BASIS_SIZE];
	cblas_scopy((MKL_INT) MONTE_CARLO_SAMPLES_NUMBER, meanPhotonNumbers, NO_INC,
					temp, NO_INC);
	cblas_sscal((MKL_INT) MONTE_CARLO_SAMPLES_NUMBER, 2.0f * meanPhotonsNumber, temp, NO_INC);
	FLOAT_TYPE sum2 = cblas_sasum((MKL_INT) MONTE_CARLO_SAMPLES_NUMBER, temp, NO_INC);

	//(a^2 + b^2 - 2 a b)
	FLOAT_TYPE sum = abs(
			sum1
					+ MONTE_CARLO_SAMPLES_NUMBER * meanPhotonsNumber
							* meanPhotonsNumber - sum2);

	FLOAT_TYPE variance = sqrtf(
			sum
					/ (MONTE_CARLO_SAMPLES_NUMBER
							* (MONTE_CARLO_SAMPLES_NUMBER - 1)));

	//free resources
	delete[] hCSR3.values;
	delete[] hCSR3.columns;
	delete[] hCSR3.rowIndex;

	delete[] aCSR3.values;
	delete[] aCSR3.columns;
	delete[] aCSR3.rowIndex;

	cout << "Mean photons number: " << meanPhotonsNumber << "\n";
	cout << "Variance: " << variance << endl;

	auto end = chrono::steady_clock::now();
	auto diff = end - start;
	cout << "Elapsed time is :  " << chrono::duration_cast < chrono::nanoseconds
			> (diff).count() / 1000000000.0 << "s" << endl;

	return 0;
}
