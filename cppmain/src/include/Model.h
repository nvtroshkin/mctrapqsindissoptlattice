/*
 * Model.h
 *
 *  Created on: Nov 17, 2017
 *      Author: fakesci
 */

#ifndef SRC_MODEL_H
#define SRC_MODEL_H

#include <mkl_types.h>
#include <eval-params.h>
#include <precision-definition.h>
#include <CSR3Matrix.h>
#include <string>

class Model {

	typedef COMPLEX_TYPE (Model::*CalcElemFuncP)(int i, int j) const;

	const MKL_INT atom1SSize;
	const MKL_INT field1SSize;
	const MKL_INT subs1Size;

	const MKL_INT atom2SSize;
	const MKL_INT field2SSize;
	const MKL_INT subs2Size;

	const MKL_INT basisSize;

	const FLOAT_TYPE kappa;
	const FLOAT_TYPE deltaOmega;
	const FLOAT_TYPE g;
	const FLOAT_TYPE scE;
	const FLOAT_TYPE J;

	//cache
	FLOAT_TYPE *sqrtsOfPhotonNumbers;

	CSR3Matrix *a1InCSR3;
	CSR3Matrix *a1PlusInCSR3;

	CSR3Matrix *a2InCSR3;
	CSR3Matrix *a2PlusInCSR3;
#ifdef H_SPARSE
	CSR3Matrix *lInCSR3;
#else
	COMPLEX_TYPE *l;
#endif

	//The basis vectors are enumerated flatly, |photon number>|atom state>,
	//|0>|0>, |0>|1> and etc.

	COMPLEX_TYPE a1PlusComplex(int i, int j) const;

	COMPLEX_TYPE a1Complex(int i, int j) const;

	COMPLEX_TYPE a2PlusComplex(int i, int j) const;

	COMPLEX_TYPE a2Complex(int i, int j) const;

	CSR3Matrix *createCSR3Matrix(CalcElemFuncP f, std::string matrixName) const;

	COMPLEX_TYPE *createMatrix(CalcElemFuncP f, std::string matrixName) const;

	CSR3Matrix *createA1InCSR3();
	CSR3Matrix *createA1PlusInCSR3();

	CSR3Matrix *createA2InCSR3();
	CSR3Matrix *createA2PlusInCSR3();

	CSR3Matrix *createLInCSR3();

public:
	Model(MKL_INT atom1SSize, MKL_INT atom2SSize, MKL_INT field1SSize,
	MKL_INT field2SSize,
	FLOAT_TYPE kappa, FLOAT_TYPE deltaOmega, FLOAT_TYPE g,
	FLOAT_TYPE scE, FLOAT_TYPE J);

	~Model();

	//------------  Internal - for tests   ----------------------------------------

	FLOAT_TYPE a1Plus(int i, int j) const;

	FLOAT_TYPE a1(int i, int j) const;

	FLOAT_TYPE sigma1Plus(int i, int j) const;

	FLOAT_TYPE sigma1Minus(int i, int j) const;

	FLOAT_TYPE a2Plus(int i, int j) const;

	FLOAT_TYPE a2(int i, int j) const;

	FLOAT_TYPE sigma2Plus(int i, int j) const;

	FLOAT_TYPE sigma2Minus(int i, int j) const;

	/**
	 * -i*Hhat, Hhat is defined by exp. (7.11), Petruchionne p363
	 *
	 * L = -i*H - kappa*a1Plus.a1 - kappa*a2Plus.a2
	 *
	 * real = -kappa*a1Plus.a1-kappa*a2Plus.a2
	 * imagine = -H
	 */
	COMPLEX_TYPE L(int i, int j) const;

	/**
	 * The Hamiltonian of the system
	 */
	FLOAT_TYPE H(int i, int j) const;

	//---------------------------------------------------------

	/**
	 * A CSR3 representation of the A1 operator
	 */
	CSR3Matrix *getA1InCSR3() const;

	/**
	 * A CSR3 representation of the A1+ operator
	 */
	CSR3Matrix *getA1PlusInCSR3() const;

	/**
	 * A CSR3 representation of the A2 operator
	 */
	CSR3Matrix *getA2InCSR3() const;

	/**
	 * A CSR3 representation of the A2+ operator
	 */
	CSR3Matrix *getA2PlusInCSR3() const;

#ifdef H_SPARSE
	/**
	 * A CSR3 representation of the Shroedinger's equation right part operator (L):
	 * dPsi/dt = L Psi.
	 */
	CSR3Matrix *getLInCSR3() const;
#else
	/**
	 * Matrix representation of the Shroedinger's equation right part operator (L):
	 * dPsi/dt = L Psi.
	 */
	COMPLEX_TYPE *getL() const;
#endif

	/**
	 * Returns the photon number of the field in the 1st cavity
	 * corresponding to a state
	 */
	int n1(int stateIndex) const;

	/**
	 * Returns the atom 1 level number corresponding to a state
	 */
	int s1(int stateIndex) const;

	/**
	 * Returns the photon number of the field in the 2nd cavity
	 * corresponding to a state
	 */
	int n2(int stateIndex) const;

	/**
	 * Returns the atom 2 level number corresponding to a state
	 */
	int s2(int stateIndex) const;

	/**
	 * Returns size of the whole basis
	 */
	MKL_INT getBasisSize() const;
};

//---------------------------------------------------------

inline CSR3Matrix *Model::getA1InCSR3() const {
	return a1InCSR3;
}

inline CSR3Matrix *Model::getA1PlusInCSR3() const {
	return a1PlusInCSR3;
}

inline CSR3Matrix *Model::getA2InCSR3() const {
	return a2InCSR3;
}

inline CSR3Matrix *Model::getA2PlusInCSR3() const {
	return a2PlusInCSR3;
}

#ifdef H_SPARSE
inline CSR3Matrix *Model::getLInCSR3() const {
	return lInCSR3;
}
#else
inline COMPLEX_TYPE *Model::getL() const {
	return l;
}
#endif

inline int Model::n1(int stateIndex) const {
	return stateIndex / (atom1SSize * subs2Size);
}

inline int Model::s1(int stateIndex) const {
	return (stateIndex / subs2Size) % atom1SSize;
}

inline int Model::n2(int stateIndex) const {
	return (stateIndex % subs1Size) / atom2SSize;
}

inline int Model::s2(int stateIndex) const {
	return (stateIndex % subs1Size) % atom2SSize;
}

inline MKL_INT Model::getBasisSize() const {
	return basisSize;
}

#endif /* SRC_MODEL_H */
