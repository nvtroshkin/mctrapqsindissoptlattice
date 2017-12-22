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

class Model {

	typedef CUDA_COMPLEX_TYPE (Model::*CalcElemFuncP)(uint i, uint j) const;

	const uint atom1SSize;
	const uint field1SSize;
	const uint subs1Size;

	const uint atom2SSize;
	const uint field2SSize;
	const uint subs2Size;

	const uint atom3SSize;
	const uint field3SSize;
	const uint subs3Size;

	const uint basisSize;

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

	CSR3Matrix *a3InCSR3;
	CSR3Matrix *a3PlusInCSR3;
#ifdef L_SPARSE
	CSR3Matrix *lInCSR3;
#else
	CUDA_COMPLEX_TYPE *l;
#endif

	//The basis vectors are enumerated flatly, |photon number>|atom state>,
	//|0>|0>, |0>|1> and etc.

	CUDA_COMPLEX_TYPE a1PlusComplex(uint i, uint j) const;

	CUDA_COMPLEX_TYPE a1Complex(uint i, uint j) const;

	CUDA_COMPLEX_TYPE a2PlusComplex(uint i, uint j) const;

	CUDA_COMPLEX_TYPE a2Complex(uint i, uint j) const;

	CUDA_COMPLEX_TYPE a3PlusComplex(uint i, uint j) const;

	CUDA_COMPLEX_TYPE a3Complex(uint i, uint j) const;

	CSR3Matrix *createCSR3Matrix(CalcElemFuncP f, const char * matrixName) const;

	CUDA_COMPLEX_TYPE *createMatrix(CalcElemFuncP f, const char * matrixName) const;

	CSR3Matrix *createA1InCSR3();
	CSR3Matrix *createA1PlusInCSR3();

	CSR3Matrix *createA2InCSR3();
	CSR3Matrix *createA2PlusInCSR3();

	CSR3Matrix *createA3InCSR3();
	CSR3Matrix *createA3PlusInCSR3();

	CSR3Matrix *createLInCSR3();

public:
	Model(uint atom1SSize, uint atom2SSize, uint atom3SSize,
	uint field1SSize,
	uint field2SSize, uint field3SSize,
	FLOAT_TYPE kappa, FLOAT_TYPE deltaOmega, FLOAT_TYPE g,
	FLOAT_TYPE scE, FLOAT_TYPE J);

	~Model();

	//------------  Internal - for tests   ----------------------------------------

	FLOAT_TYPE a1Plus(uint i, uint j) const;

	FLOAT_TYPE a1(uint i, uint j) const;

	FLOAT_TYPE sigma1Plus(uint i, uint j) const;

	FLOAT_TYPE sigma1Minus(uint i, uint j) const;

	FLOAT_TYPE a2Plus(uint i, uint j) const;

	FLOAT_TYPE a2(uint i, uint j) const;

	FLOAT_TYPE sigma2Plus(uint i, uint j) const;

	FLOAT_TYPE sigma2Minus(uint i, uint j) const;

	FLOAT_TYPE a3Plus(uint i, uint j) const;

	FLOAT_TYPE a3(uint i, uint j) const;

	FLOAT_TYPE sigma3Plus(uint i, uint j) const;

	FLOAT_TYPE sigma3Minus(uint i, uint j) const;

	/**
	 * -i*Hhat, Hhat is defined by exp. (7.11), Petruchionne p363
	 *
	 * L = -i*H - kappa*a1Plus.a1 - kappa*a2Plus.a2
	 *
	 * real = -kappa*a1Plus.a1-kappa*a2Plus.a2
	 * imagine = -H
	 */
	CUDA_COMPLEX_TYPE L(uint i, uint j) const;

	/**
	 * The Hamiltonian of the system
	 */
	FLOAT_TYPE H(uint i, uint j) const;

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

	/**
	 * A CSR3 representation of the A3 operator
	 */
	CSR3Matrix *getA3InCSR3() const;

	/**
	 * A CSR3 representation of the A3+ operator
	 */
	CSR3Matrix *getA3PlusInCSR3() const;

#ifdef L_SPARSE
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
	CUDA_COMPLEX_TYPE *getL() const;
#endif

	/**
	 * Returns the photon number of the field in the 1st cavity
	 * corresponding to a state
	 */
	uint n1(uint stateIndex) const;

	/**
	 * Returns the atom 1 level number corresponding to a state
	 */
	uint s1(uint stateIndex) const;

	/**
	 * Returns the photon number of the field in the 2nd cavity
	 * corresponding to a state
	 */
	uint n2(uint stateIndex) const;

	/**
	 * Returns the atom 2 level number corresponding to a state
	 */
	uint s2(uint stateIndex) const;

	/**
	 * Returns the photon number of the field in the 3rd cavity
	 * corresponding to a state
	 */
	uint n3(uint stateIndex) const;

	/**
	 * Returns the atom 3 level number corresponding to a state
	 */
	uint s3(uint stateIndex) const;

	/**
	 * Returns size of the whole basis
	 */
	uint getBasisSize() const;
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

inline CSR3Matrix *Model::getA3InCSR3() const {
	return a3InCSR3;
}

inline CSR3Matrix *Model::getA3PlusInCSR3() const {
	return a3PlusInCSR3;
}

#ifdef L_SPARSE
inline CSR3Matrix *Model::getLInCSR3() const {
	return lInCSR3;
}
#else
inline CUDA_COMPLEX_TYPE *Model::getL() const {
	return l;
}
#endif

inline uint Model::n1(uint stateIndex) const {
	return stateIndex / (atom1SSize * subs2Size * subs3Size);
}

inline uint Model::s1(uint stateIndex) const {
	return (stateIndex / (subs2Size * subs3Size)) % atom1SSize;
}

inline uint Model::n2(uint stateIndex) const {
	return (stateIndex / (atom2SSize * subs3Size)) % field2SSize;
}

inline uint Model::s2(uint stateIndex) const {
	return (stateIndex / subs3Size) % atom2SSize;
}

inline uint Model::n3(uint stateIndex) const {
	return (stateIndex / atom3SSize) % field3SSize;
}

inline uint Model::s3(uint stateIndex) const {
	return stateIndex % atom3SSize;
}

inline uint Model::getBasisSize() const {
	return basisSize;
}

#endif /* SRC_MODEL_H */
