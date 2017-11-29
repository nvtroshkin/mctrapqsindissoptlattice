/*
 *  Created on: Nov 18, 2017
 *      Author: fake_sci
 */

#include <iostream>
#include <precision-definition.h>
#include <CSR3Matrix.h>
#include <utilities.h>

#include <ModelBuilder.h>

ModelBuilder::ModelBuilder(int maxPhotonNumber, MKL_INT basisSize,
		FLOAT_TYPE kappa, FLOAT_TYPE deltaOmega, FLOAT_TYPE g,
		FLOAT_TYPE latinE) :
		basisSize(basisSize), kappa(kappa), deltaOmega(deltaOmega), g(g), latinE(
				latinE) {
	sqrtsOfPhotonNumbers = new FLOAT_TYPE[maxPhotonNumber + 1]; //plus a zero photons state
	FLOAT_TYPE photonNumbers[maxPhotonNumber + 1];
	for (int k = 0; k < maxPhotonNumber + 1; k++) {
		photonNumbers[k] = k;
	}
	vSqrt((MKL_INT) (maxPhotonNumber + 1), photonNumbers,
			sqrtsOfPhotonNumbers);

	aInCSR3 = createAInCSR3();
	aPlusInCSR3 = createAPlusInCSR3();
	hInCSR3 = createHInCSR3();
}

ModelBuilder::~ModelBuilder() {
	delete[] sqrtsOfPhotonNumbers;
	delete aInCSR3;
	delete aPlusInCSR3;
	delete hInCSR3;
}

//to obtain a just swap i and j
FLOAT_TYPE ModelBuilder::aPlus(int i, int j) {
	if (n(i) != n(j) + 1 || s(i) != s(j)) {
		return 0.0f;
	}

	return sqrtsOfPhotonNumbers[n(j) + 1];
}

//to obtain sigmaMinus just swap i and j
FLOAT_TYPE ModelBuilder::sigmaPlus(int i, int j) {
	if (s(j) != 0 || s(i) != 1 || n(i) != n(j)) {
		return 0.0f;
	}

	return 1.0f;
}

inline COMPLEX_TYPE ModelBuilder::H(int i, int j) {
	//the real part of the matrix element
	FLOAT_TYPE imaginary = 0.0f;
	for (int k = 0; k < basisSize; k++) {
		imaginary -=
				-deltaOmega
						* (aPlus(i, k) * aPlus(j, k)
								+ sigmaPlus(i, k) * sigmaPlus(j, k))
						+ g
								* (aPlus(k, i) * sigmaPlus(k, j)
										+ aPlus(i, k) * sigmaPlus(j, k));
	}

	imaginary -= latinE * (aPlus(i, j) + aPlus(j, i));

	//the imaginary
	FLOAT_TYPE real = 0.0f;
	for (int k = 0; k < basisSize; k++) {
		real -= aPlus(i, k) * aPlus(j, k);
	}

	real *= kappa;

	return {real, imaginary};
}

inline CSR3Matrix *ModelBuilder::createAPlusInCSR3() {
	//aPlus is a 1-diagonal matrix

	//1 - the number of diagonals, 2 - the cut off corner elements
	// - 0000
	//  -0000
	//   1000
	//   0100

	CSR3Matrix *csr3Matrix = new CSR3Matrix(basisSize, basisSize);

	const int vertOffset = 2;

	int colNum;
	int currValueIndex = -1;	//to hold current index
	int i;
	for (i = 0; i < basisSize; i++) {
		colNum = i - vertOffset;
		currValueIndex++;
		if (colNum >= 0) {
			csr3Matrix->values[currValueIndex] = {aPlus(i, colNum),0};
			csr3Matrix->columns[currValueIndex] = colNum;
		} else {
			csr3Matrix->values[currValueIndex] = {0,0};//the first two rows - by zero
			csr3Matrix->columns[currValueIndex] = 0;
		}
		csr3Matrix->rowIndex[i] = currValueIndex;	//each value on its own row
	}

	return csr3Matrix;
}

inline CSR3Matrix *ModelBuilder::createAInCSR3() {
	//a is a 1-diagonal matrix

	//1 - the number of diagonals, 2 - the cut off corner elements
	// 0010
	// 0001
	// 0000-
	// 0000 -

	CSR3Matrix *csr3Matrix = new CSR3Matrix(basisSize, basisSize);

	const int tailPadding = 2;//additional zero elements in place of zero rows at the bottom

	int colNum;
	int currValueIndex = -1;	//to hold current index
	int i;
	for (i = 0; i < basisSize; i++) {
		colNum = i + tailPadding;
		currValueIndex++;
		if (colNum < basisSize) {
			csr3Matrix->values[currValueIndex] = {aPlus(colNum, i),0}; //swap arguments
			csr3Matrix->columns[currValueIndex] = colNum;
		} else {
			csr3Matrix->values[currValueIndex] = {0,0}; //the first two rows - by zero
			csr3Matrix->columns[currValueIndex] = basisSize - 1;
		}
		csr3Matrix->rowIndex[i] = currValueIndex;	//each value on its own row
	}

	return csr3Matrix;
}

inline CSR3Matrix *ModelBuilder::createHInCSR3() {
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

	CSR3Matrix *csr3Matrix = new CSR3Matrix(basisSize, basisSize * diagsNumber - 6);

	int colNum;
	int currValueIndex = -1;
	int i;
	for (i = 0; i < basisSize; i++) {
		for (int j = 0; j < diagsNumber; j++) {
			colNum = i - halfDiagsNumber + j;
			if (colNum >= 0 && colNum < basisSize) {
				currValueIndex++;
				csr3Matrix->values[currValueIndex] = H(i, colNum);
				csr3Matrix->columns[currValueIndex] = colNum;
			}
		}

		if (i < halfDiagsNumber) {
			csr3Matrix->rowIndex[i] = currValueIndex - halfDiagsNumber - i;
		} else if (i >= basisSize - halfDiagsNumber) {
			csr3Matrix->rowIndex[i] = currValueIndex - halfDiagsNumber + 1
					- basisSize + i;
		} else {
			csr3Matrix->rowIndex[i] = currValueIndex - diagsNumber + 1;
		}
	}

	return csr3Matrix;
}

//MatrixDiagForm getHhatInDiagForm() {
//
//	MatrixDiagForm diagH;
//
//	//Hhat is a 5-diagonal symmetrical matrix
//	//it can be stored by calculating the main and two lower diagonals
//	//and stored into the diagonal matrix storage format (https://software.intel.com/en-us/mkl-developer-reference-c-sparse-blas-diagonal-matrix-storage-format)
//
//	//diagonals in relation to the main diagonal
//	//only 5
//	diagH.diagDistLength = 5;
//	diagH.leadDimension = basisSize;
//	diagH.diagsDistances = new int[5] { -2, -1, 0, 1, 2 };
//	//the compact diagonals storage
//	diagH.matrix = new COMPLEX_TYPE[basisSize * 5];
//
//	//Part of the indices goes out of the boundaries, at the beginning abd at the end
//	//
//	// --xxx00
//	//  -xxxx0
//	//   xxxxx
//	//   0xxxx-
//	//   00xxx--
//	for (int i = 0; i < basisSize; i++) {
//		for (int j = 0; j < diagH.diagDistLength; j++) {
//			diagH.matrix[i * diagH.diagDistLength + j] = H(i, i + j - 2,
//					basisSize, KAPPA, DELTA_OMEGA, G, LATIN_E);
//		}
//	}
//
//	return diagH;
//}
