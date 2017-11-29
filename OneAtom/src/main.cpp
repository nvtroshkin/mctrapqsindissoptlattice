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
#include <mkl-constants.h>
#include <ModelBuilder.h>
#include <system-constants.h>
#include <Solver.h>

using namespace std;

int main(int argc, char **argv) {
	auto start = chrono::steady_clock::now();

	//The basis vectors are enumerated flatly, |photon number>|atom state>, |0>|0>, |0>|1> and etc.

	//init cache
	COMPLEX_TYPE zeroVector[DRESSED_BASIS_SIZE];
	for (int i = 0; i < DRESSED_BASIS_SIZE; i++) {
		zeroVector[i].real = 0.0f;
		zeroVector[i].imag = 0.0f;
	}

	ModelBuilder modelBuilder(MAX_PHOTON_NUMBER, DRESSED_BASIS_SIZE, KAPPA,
			DELTA_OMEGA, G, LATIN_E);

	const CSR3Matrix * const hCSR3 = modelBuilder.getHInCSR3();
	const int *HCSR3RowIndex = hCSR3->rowIndex;
	const int *HCSR3Columns = hCSR3->columns;
	const COMPLEX_TYPE *HCSR3Values = hCSR3->values;

	const CSR3Matrix * const aCSR3 = modelBuilder.getAInCSR3();
	const int *aCSR3RowIndex = aCSR3->rowIndex;
	const int *aCSR3Columns = aCSR3->columns;
	const COMPLEX_TYPE *aCSR3Values = aCSR3->values;

	//A storage of final states of all realizations
	COMPLEX_TYPE result[MONTE_CARLO_SAMPLES_NUMBER][DRESSED_BASIS_SIZE];
	//the previous step vector
	COMPLEX_TYPE step1State[DRESSED_BASIS_SIZE];
	//the currently being calculated vector
	COMPLEX_TYPE step2State[DRESSED_BASIS_SIZE];

	Solver solver((MKL_INT) DRESSED_BASIS_SIZE);

//#pragma omp parallel for
	for (int sampleIndex = 0; sampleIndex < MONTE_CARLO_SAMPLES_NUMBER;
			sampleIndex++) {
		//Initialize each sample by the ground state vector
		cblas_ccopy((MKL_INT) DRESSED_BASIS_SIZE, zeroVector, NO_INC,
				step1State, NO_INC);
		step1State[0].real = 1.0f; //the ground state

		solver.solve(HCSR3Values, HCSR3RowIndex, HCSR3Columns, aCSR3Values,
				aCSR3RowIndex, aCSR3Columns, step1State, step2State);

		//store
		cblas_ccopy((MKL_INT) DRESSED_BASIS_SIZE, step2State, NO_INC,
				result[sampleIndex], NO_INC);
	}

	//now the Monte-Carlo simulations are over
	//we should calculate the mean photons number and the variance

	//an auxiliary array with photon numbers for each basis vector is needed
	COMPLEX_TYPE statePhotonNumber[DRESSED_BASIS_SIZE];
	for (int i = 0; i < DRESSED_BASIS_SIZE; i++) {
		statePhotonNumber[i] = {(FLOAT_TYPE)modelBuilder.n(i),0.0f};
	}

	//Sum(<psi|n|psi>)
	//mult psi on ns
	FLOAT_TYPE meanPhotonNumbers[MONTE_CARLO_SAMPLES_NUMBER];
	COMPLEX_TYPE norm2;
	COMPLEX_TYPE tempVector[DRESSED_BASIS_SIZE];
	for (int i = 0; i < MONTE_CARLO_SAMPLES_NUMBER; i++) {
		vcMul((MKL_INT) DRESSED_BASIS_SIZE, result[i], statePhotonNumber,
				tempVector);
		cblas_cdotc_sub((MKL_INT) DRESSED_BASIS_SIZE, tempVector, NO_INC,
				result[i], NO_INC, &norm2);

		//store for the variance
		meanPhotonNumbers[i] = norm2.real;
	}

	FLOAT_TYPE meanPhotonsNumber = cblas_sasum(
			(MKL_INT) MONTE_CARLO_SAMPLES_NUMBER, meanPhotonNumbers, NO_INC);
	meanPhotonsNumber /= MONTE_CARLO_SAMPLES_NUMBER;

	//variance. Calculate like this to avoid close numbers subtraction
	//Sum(mean photon numbers)^2
	FLOAT_TYPE sum1 = cblas_sdsdot((MKL_INT) MONTE_CARLO_SAMPLES_NUMBER, 0.0f,
			meanPhotonNumbers, NO_INC, meanPhotonNumbers, NO_INC);

	//Sum(2*mean photon number*mean photon number[i])
	FLOAT_TYPE temp[DRESSED_BASIS_SIZE];
	cblas_scopy((MKL_INT) MONTE_CARLO_SAMPLES_NUMBER, meanPhotonNumbers, NO_INC,
			temp, NO_INC);
	cblas_sscal((MKL_INT) MONTE_CARLO_SAMPLES_NUMBER, 2.0f * meanPhotonsNumber,
			temp, NO_INC);
	FLOAT_TYPE sum2 = cblas_sasum((MKL_INT) MONTE_CARLO_SAMPLES_NUMBER, temp,
			NO_INC);

	//(a^2 + b^2 - 2 a b)
	FLOAT_TYPE sum = abs(
			sum1
					+ MONTE_CARLO_SAMPLES_NUMBER * meanPhotonsNumber
							* meanPhotonsNumber - sum2);

	FLOAT_TYPE variance = sqrtf(
			sum
					/ (MONTE_CARLO_SAMPLES_NUMBER
							* (MONTE_CARLO_SAMPLES_NUMBER - 1)));

	cout << "Mean photons number: " << meanPhotonsNumber << "\n";
	cout << "Variance: " << variance << endl;

	auto end = chrono::steady_clock::now();
	auto diff = end - start;
	cout << "Elapsed time is :  " << chrono::duration_cast < chrono::nanoseconds
			> (diff).count() / 1000000000.0 << "s" << endl;

	return 0;
}
