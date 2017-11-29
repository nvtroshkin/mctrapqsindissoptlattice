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
#include <CSR3Matrix.h>

//struct MatrixDiagForm {
//	int diagDistLength;
//	int leadDimension;
//	int *diagsDistances;
//	//
//	COMPLEX_TYPE *matrix;
//};

class ModelBuilder {
	const MKL_INT basisSize;

	const FLOAT_TYPE kappa;
	const FLOAT_TYPE deltaOmega;
	const FLOAT_TYPE g;
	const FLOAT_TYPE latinE;

	//cache
	FLOAT_TYPE *sqrtsOfPhotonNumbers;
	CSR3Matrix *aInCSR3;
	CSR3Matrix *aPlusInCSR3;
	CSR3Matrix *hInCSR3;

	//The basis vectors are enumerated flatly, |photon number>|atom state>, |0>|0>, |0>|1> and etc.
	CSR3Matrix *createAInCSR3();
	CSR3Matrix *createAPlusInCSR3();
	CSR3Matrix *createHInCSR3();

public:
	ModelBuilder(int maxPhotonNumber, MKL_INT basisSize, FLOAT_TYPE kappa,
	FLOAT_TYPE deltaOmega, FLOAT_TYPE g, FLOAT_TYPE latinE);
	~ModelBuilder();

	FLOAT_TYPE aPlus(int i, int j);
	//
	FLOAT_TYPE sigmaPlus(int i, int j);
	//The Hhat from Petruchionne p363, the (7.11) expression
	COMPLEX_TYPE H(int i, int j);

	//getters and setters
	CSR3Matrix *getAInCSR3() const {
		return aInCSR3;
	}

	CSR3Matrix *getAPlusInCSR3() const {
		return aPlusInCSR3;
	}

	CSR3Matrix *getHInCSR3() const {
		return hInCSR3;
	}
};

#endif /* SRC_MODEL_BUILDER_H */
