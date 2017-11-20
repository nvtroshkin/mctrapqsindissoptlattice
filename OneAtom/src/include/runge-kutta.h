/*
 * main.h
 *
 *  Created on: Nov 17, 2017
 *      Author: fake_sci
 */

#ifndef SRC_RUNGE_KUTTA_H_
#define SRC_RUNGE_KUTTA_H_

#include <mkl.h>

struct MatrixDiagForm {
	int diagDistLength;
	int *diagsDistances;
	MKL_Complex8 *matrix;
};

void initPhotonNumbersSqrts();
float aPlus(int i, int j);
float sigmaPlus(int i, int j);
MatrixDiagForm getHhatInDiagForm();

//inline functions
inline int n(int index) {
	return index / 2;
}

inline int s(int index) {
	return index % 2;
}

//hbar = 1
//The Hhat from Petruchionne p363, the (7.11) expression
inline MKL_Complex8 H(int i, int j, int DRESSED_BASIS_SIZE, float KAPPA,
		float DELTA_OMEGA, float G, float LATIN_E) {
	//the real part of the matrix element
	float imaginary = 0.0f;
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
	float real = 0.0;
	for (int k = 0; k < DRESSED_BASIS_SIZE; k++) {
		real -= aPlus(i, k) * aPlus(j, k);
	}

	real *= KAPPA;

	return {real, imaginary};
}

#endif /* SRC_RUNGE_KUTTA_H_ */
