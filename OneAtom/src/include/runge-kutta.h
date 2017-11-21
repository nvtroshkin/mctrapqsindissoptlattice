/*
 * main.h
 *
 *  Created on: Nov 17, 2017
 *      Author: fake_sci
 */

#ifndef SRC_RUNGE_KUTTA_H_
#define SRC_RUNGE_KUTTA_H_

#include <mkl.h>
#include <eval-params.h>

struct MatrixDiagForm {
	int diagDistLength;
	int leadDimension;
	int *diagsDistances;
	MKL_Complex8 *matrix;
};

struct CSR3Matrix {
	int rowsNumber;
	MKL_Complex8 *values;
	int *columns;
	int *rowIndex;	//indices from values of the first
					//non-null row elements of the matrix being compressed
					//the last element - total number of elements in values
};

void initPhotonNumbersSqrts();
float aPlus(int i, int j);
float sigmaPlus(int i, int j);
MatrixDiagForm getHhatInDiagForm();
CSR3Matrix getHInCSR3();
CSR3Matrix getAPlusInCSR3();
CSR3Matrix getAInCSR3();

inline MKL_Complex8 H(int i, int j, int DRESSED_BASIS_SIZE, float KAPPA,
		float DELTA_OMEGA, float G, float LATIN_E);

//inline functions
inline int n(int index) {
	return index / 2;
}

inline int s(int index) {
	return index % 2;
}

#endif /* SRC_RUNGE_KUTTA_H_ */
