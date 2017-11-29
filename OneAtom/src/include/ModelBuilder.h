/*
 * main.h
 *
 *  Created on: Nov 17, 2017
 *      Author: fake_sci
 */

#ifndef SRC_MODEL_BUILDER_H
#define SRC_MODEL_BUILDER_H

#include <mkl_types.h>
#include <eval-params.h>
#include <precision-definition.h>

//struct MatrixDiagForm {
//	int diagDistLength;
//	int leadDimension;
//	int *diagsDistances;
//	//
//	COMPLEX_TYPE *matrix;
//};

struct CSR3Matrix {

	int rowsNumber;
	int nonZeroValuesNumber;
	//
	COMPLEX_TYPE *values;
	int *columns;
	int *rowIndex;	//indices from values of the first
	//non-null row elements of the matrix being compressed
	//the last element - total number of elements in values

	CSR3Matrix(int rowsNumber, int nonZeroValuesNumber) :
			rowsNumber(rowsNumber), nonZeroValuesNumber(nonZeroValuesNumber), values(
					new COMPLEX_TYPE[nonZeroValuesNumber]), columns(
					new int[nonZeroValuesNumber]), rowIndex(
					new int[rowsNumber + 1]/*non-zero element on each row*/) {
		//put the length of the values array at the end
		rowIndex[rowsNumber] = nonZeroValuesNumber;
	}

	~CSR3Matrix() {
		delete[] values;
		delete[] columns;
		delete[] rowIndex;
	}
};

class ModelBuilder {
	const MKL_INT BASIS_SIZE;

	const FLOAT_TYPE KAPPA;
	const FLOAT_TYPE DELTA_OMEGA;
	const FLOAT_TYPE G;
	const FLOAT_TYPE LATIN_E;

	//cache
	FLOAT_TYPE *sqrtsOfPhotonNumbers;
	CSR3Matrix *A_IN_CSR3;
	CSR3Matrix *A_PLUS_IN_CSR3;
	CSR3Matrix *H_IN_CSR3;

	CSR3Matrix *createAInCSR3();
	CSR3Matrix *createAPlusInCSR3();
	CSR3Matrix *createHInCSR3();

public:
	ModelBuilder(int maxPhotonNumber, MKL_INT basisSize, FLOAT_TYPE kappa,
	FLOAT_TYPE deltaOmega, FLOAT_TYPE g, FLOAT_TYPE latinE);
	~ModelBuilder();

	//inline functions
	int n(int index) {
		return index / 2;
	}
	int s(int index) {
		return index % 2;
	}

	FLOAT_TYPE aPlus(int i, int j);
	//
	FLOAT_TYPE sigmaPlus(int i, int j);
	//The Hhat from Petruchionne p363, the (7.11) expression
	COMPLEX_TYPE H(int i, int j);

	//getters and setters
	CSR3Matrix *getAInCSR3() const {
		return A_IN_CSR3;
	}

	CSR3Matrix *getAPlusInCSR3() const {
		return A_PLUS_IN_CSR3;
	}

	CSR3Matrix *getHInCSR3() const {
		return H_IN_CSR3;
	}
};

#endif /* SRC_MODEL_BUILDER_H */
