/*
 * runge-kutta.cpp
 *
 *  Created on: Nov 18, 2017
 *      Author: fake_sci
 */

#include <runge-kutta.h>

#include <iostream>
#include <mkl.h>
#include <system-constants.h>
#include <eval-params.h>	//probably, should not be here

static float sqrtsOfPhotonNumbers[MAX_PHOTON_NUMBER + 1];

//Please, do not forget to call this before usage (Should be a class with a constructor?)
void initPhotonNumbersSqrts() {
	//calculate square roots
	float photonNumbers[MAX_PHOTON_NUMBER + 1];
	for (int k = 0; k < MAX_PHOTON_NUMBER + 1; k++) {
		photonNumbers[k] = k;
	}

	vsSqrt((MKL_INT) (MAX_PHOTON_NUMBER + 1), photonNumbers,
			sqrtsOfPhotonNumbers);
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

MatrixDiagForm getHhatInDiagForm() {

	MatrixDiagForm diagH;

	//Hhat is a 5-diagonal symmetrical matrix
	//it can be stored by calculating the main and two lower diagonals
	//and stored into the diagonal matrix storage format (https://software.intel.com/en-us/mkl-developer-reference-c-sparse-blas-diagonal-matrix-storage-format)

	//diagonals in relation to the main diagonal
	//only 5
	diagH.diagDistLength = 5;
	diagH.diagsDistances = new int[5]{ -2, -1, 0, 1, 2 };
	//the compact diagonals storage
	diagH.matrix = new MKL_Complex8[DRESSED_BASIS_SIZE * 5];

	//Part of the indices goes out of the boundaries, at the beginning abd at the end
	//
	// --xxx00
	//  -xxxx0
	//   xxxxx
	//   0xxxx-
	//   00xxx--
	for (int i = 0; i < DRESSED_BASIS_SIZE; i++) {
		for (int j = 0; j < diagH.diagDistLength; j++) {
			diagH.matrix[i * diagH.diagDistLength + j] = H(i, i + j - 2,
					DRESSED_BASIS_SIZE, KAPPA, DELTA_OMEGA, G, LATIN_E);
		}
	}

	return diagH;
}
