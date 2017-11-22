/*
 * runge-kutta.cpp
 *
 *  Created on: Nov 18, 2017
 *      Author: fake_sci
 */

#include <macro.h>
#include <runge-kutta.h>

#include <iostream>
#include <mkl.h>
#include <system-constants.h>
#include <eval-params.h>	//probably, should not be here

static FLOAT_TYPE sqrtsOfPhotonNumbers[MAX_PHOTON_NUMBER + 1];

//Please, do not forget to call this before usage (Should be a class with a constructor?)
void initPhotonNumbersSqrts() {
	//calculate square roots
	FLOAT_TYPE photonNumbers[MAX_PHOTON_NUMBER + 1];
	for (int k = 0; k < MAX_PHOTON_NUMBER + 1; k++) {
		photonNumbers[k] = k;
	}

	vsSqrt((MKL_INT) (MAX_PHOTON_NUMBER + 1), photonNumbers,
			sqrtsOfPhotonNumbers);
}

//to obtain a just swap i and j
FLOAT_TYPE aPlus(int i, int j) {
	if (n(i) != n(j) + 1 || s(i) != s(j)) {
		return 0.0f;
	}

	return sqrtsOfPhotonNumbers[n(j) + 1];
}

//to obtain sigmaMinus just swap i and j
FLOAT_TYPE sigmaPlus(int i, int j) {
	if (s(j) != 0 || s(i) != 1 || n(i) != n(j)) {
		return 0.0f;
	}

	return 1.0f;
}

//hbar = 1
//The Hhat from Petruchionne p363, the (7.11) expression
inline COMPLEX_TYPE H(int i, int j, int DRESSED_BASIS_SIZE, FLOAT_TYPE KAPPA,
		FLOAT_TYPE DELTA_OMEGA, FLOAT_TYPE G, FLOAT_TYPE LATIN_E) {
	//the real part of the matrix element
	FLOAT_TYPE imaginary = 0.0f;
	for (int k = 0; k < DRESSED_BASIS_SIZE; k++) {
		imaginary -=
				-DELTA_OMEGA
						* (aPlus(i, k) * aPlus(j, k)
								+ sigmaPlus(i, k) * sigmaPlus(j, k))
						+ G
								* (aPlus(k, i) * sigmaPlus(k, j)
										+ aPlus(i, k) * sigmaPlus(j, k));
	}

	imaginary -= LATIN_E * (aPlus(i, j) + aPlus(j, i));

	//the imaginary
	FLOAT_TYPE real = 0.0f;
	for (int k = 0; k < DRESSED_BASIS_SIZE; k++) {
		real -= aPlus(i, k) * aPlus(j, k);
	}

	real *= KAPPA;

	return {real, imaginary};
}

CSR3Matrix getAPlusInCSR3() {
	CSR3Matrix csr3Matrix;

	//aPlus is a 1-diagonal matrix

	//1 - the number of diagonals, 2 - the cut off corner elements
	// - 0000
	//  -0000
	//   1000
	//   0100
	const int vertOffset = 2;
	csr3Matrix.rowsNumber = DRESSED_BASIS_SIZE;
	csr3Matrix.values = new COMPLEX_TYPE[DRESSED_BASIS_SIZE];
	csr3Matrix.columns = new int[DRESSED_BASIS_SIZE];
	csr3Matrix.rowIndex = new int[DRESSED_BASIS_SIZE+1]; //non-zero elements on each row

	int colNum;
	int currValueIndex = -1;	//to hold current index
	int i;
	for (i = 0; i < DRESSED_BASIS_SIZE; i++) {
		colNum = i - vertOffset;
		currValueIndex++;
		if (colNum >= 0) {
			csr3Matrix.values[currValueIndex] = {aPlus(i, colNum),0};
			csr3Matrix.columns[currValueIndex] = colNum;
		} else {
			csr3Matrix.values[currValueIndex] = {0,0};	//the first two rows - by zero
			csr3Matrix.columns[currValueIndex] = 0;
		}
		csr3Matrix.rowIndex[i] = currValueIndex;	//each value on its own row
	}

	csr3Matrix.rowIndex[i] = currValueIndex + 1; //the length of the values array

	return csr3Matrix;
}

CSR3Matrix getAInCSR3() {
	CSR3Matrix csr3Matrix;

	//a is a 1-diagonal matrix

	//1 - the number of diagonals, 2 - the cut off corner elements
	// 0010
	// 0001
	// 0000-
	// 0000 -
	const int tailPadding = 2;	//additional zero elements in place of zero rows at the bottom
	csr3Matrix.rowsNumber = DRESSED_BASIS_SIZE;
	csr3Matrix.values = new COMPLEX_TYPE[DRESSED_BASIS_SIZE];
	csr3Matrix.columns = new int[DRESSED_BASIS_SIZE];
	csr3Matrix.rowIndex = new int[DRESSED_BASIS_SIZE+1]; //non-zero elements on each row

	int colNum;
	int currValueIndex = -1;	//to hold current index
	int i;
	for (i = 0; i < DRESSED_BASIS_SIZE; i++) {
		colNum = i + tailPadding;
		currValueIndex++;
		if (colNum < DRESSED_BASIS_SIZE) {
			csr3Matrix.values[currValueIndex] = {aPlus(colNum, i),0}; //swap arguments
			csr3Matrix.columns[currValueIndex] = colNum;
		} else {
			csr3Matrix.values[currValueIndex] = {0,0};	//the first two rows - by zero
			csr3Matrix.columns[currValueIndex] = DRESSED_BASIS_SIZE - 1;
		}
		csr3Matrix.rowIndex[i] = currValueIndex;	//each value on its own row
	}

	csr3Matrix.rowIndex[i] = currValueIndex + 1; //the length of the values array

	return csr3Matrix;
}

CSR3Matrix getHInCSR3() {

	CSR3Matrix csr3Matrix;

	//Hhat is a 5-diagonal symmetrical matrix
	//it can be stored into the CSR3 matrix storage format, evaluating only 5 diags
	//(https://software.intel.com/en-us/mkl-developer-reference-c-sparse-blas-csr-matrix-storage-format)

	//5 - the number of diagonals, 6 - the cut off corner elements
	// --xxx00
	//  -xxxx0
	//   xxxxx
	//   0xxxx-
	//   00xxx--
	const int diagsNumber = 5;
	const int halfDiagsNumber = diagsNumber / 2;
	csr3Matrix.rowsNumber = DRESSED_BASIS_SIZE;
	csr3Matrix.values = new COMPLEX_TYPE[DRESSED_BASIS_SIZE * diagsNumber - 6];
	csr3Matrix.columns = new int[DRESSED_BASIS_SIZE * diagsNumber - 6];
	csr3Matrix.rowIndex = new int[DRESSED_BASIS_SIZE+1]; //non-zero elements on each row

	int colNum;
	int currValueIndex = -1;
	int i;
	for (i = 0; i < DRESSED_BASIS_SIZE; i++) {
		for (int j = 0; j < diagsNumber; j++) {
			colNum = i - halfDiagsNumber + j;
			if (colNum >= 0 && colNum < DRESSED_BASIS_SIZE) {
				currValueIndex++;
				csr3Matrix.values[currValueIndex] = H(i, colNum,
						DRESSED_BASIS_SIZE, KAPPA, DELTA_OMEGA, G, LATIN_E);
				csr3Matrix.columns[currValueIndex] = colNum;
			}
		}

		if (i < halfDiagsNumber) {
			csr3Matrix.rowIndex[i] = currValueIndex - halfDiagsNumber - i;
		} else if (i >= DRESSED_BASIS_SIZE - halfDiagsNumber) {
			csr3Matrix.rowIndex[i] = currValueIndex - halfDiagsNumber + 1
					- DRESSED_BASIS_SIZE + i;
		} else {
			csr3Matrix.rowIndex[i] = currValueIndex - diagsNumber + 1;
		}
	}

	csr3Matrix.rowIndex[i] = currValueIndex + 1; //the length of the values array

	return csr3Matrix;
}

MatrixDiagForm getHhatInDiagForm() {

	MatrixDiagForm diagH;

	//Hhat is a 5-diagonal symmetrical matrix
	//it can be stored by calculating the main and two lower diagonals
	//and stored into the diagonal matrix storage format (https://software.intel.com/en-us/mkl-developer-reference-c-sparse-blas-diagonal-matrix-storage-format)

	//diagonals in relation to the main diagonal
	//only 5
	diagH.diagDistLength = 5;
	diagH.leadDimension = DRESSED_BASIS_SIZE;
	diagH.diagsDistances = new int[5] { -2, -1, 0, 1, 2 };
	//the compact diagonals storage
	diagH.matrix = new COMPLEX_TYPE[DRESSED_BASIS_SIZE * 5];

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
