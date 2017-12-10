/*
 *  Created on: Nov 18, 2017
 *      Author: fakesci
 */

#include <Model.h>
#include <utilities.h>
#include <precision-definition.h>
#include <mkl-constants.h>

Model::Model(MKL_INT atom1SSize, MKL_INT atom2SSize, MKL_INT field1SSize,
MKL_INT field2SSize,
FLOAT_TYPE kappa, FLOAT_TYPE deltaOmega, FLOAT_TYPE g,
FLOAT_TYPE scE, FLOAT_TYPE J) :
		atom1SSize(atom1SSize), field1SSize(field1SSize), subs1Size(
				atom1SSize * field1SSize), atom2SSize(atom2SSize), field2SSize(
				field2SSize), subs2Size(atom2SSize * field2SSize), basisSize(
				subs1Size * subs2Size), kappa(kappa), deltaOmega(deltaOmega), g(
				g), scE(scE), J(J) {
	MKL_INT maxFieldSSize = std::max(field1SSize, field2SSize);

	sqrtsOfPhotonNumbers = new FLOAT_TYPE[maxFieldSSize];
	FLOAT_TYPE photonNumbers[maxFieldSSize];
	for (int k = 0; k < maxFieldSSize; ++k) {
		photonNumbers[k] = k;
	}
	vSqrt(maxFieldSSize, photonNumbers, sqrtsOfPhotonNumbers);

	a1InCSR3 = createCSR3Matrix(&Model::a1Complex);
	a1PlusInCSR3 = createCSR3Matrix(&Model::a1PlusComplex);

	a2InCSR3 = createCSR3Matrix(&Model::a2Complex);
	a2PlusInCSR3 = createCSR3Matrix(&Model::a2PlusComplex);

#ifdef H_SPARSE
	lInCSR3 = createCSR3Matrix(&Model::L);
#else
	l = createMatrix(&Model::L);
#endif
}

Model::~Model() {
	delete[] sqrtsOfPhotonNumbers;
	delete a1InCSR3;
	delete a1PlusInCSR3;
	delete a2InCSR3;
	delete a2PlusInCSR3;

#ifdef H_SPARSE
	delete lInCSR3;
#else
	delete[] l;
#endif
}

inline FLOAT_TYPE Model::a1Plus(int i, int j) const {
	return (n1(i) != n1(j) + 1 || s1(i) != s1(j)) ?
			0.0 : sqrtsOfPhotonNumbers[n1(j) + 1];
}

inline FLOAT_TYPE Model::a1(int i, int j) const {
	return a1Plus(j, i);
}

inline FLOAT_TYPE Model::sigma1Plus(int i, int j) const {
	return (s1(j) != 0 || s1(i) != 1 || n1(i) != n1(j)) ? 0.0 : 1.0;
}

inline FLOAT_TYPE Model::sigma1Minus(int i, int j) const {
	return sigma1Plus(j, i);
}

inline FLOAT_TYPE Model::a2Plus(int i, int j) const {
	return (n2(i) != n2(j) + 1 || s2(i) != s2(j)) ?
			0.0 : sqrtsOfPhotonNumbers[n2(j) + 1];
}

inline FLOAT_TYPE Model::a2(int i, int j) const {
	return a2Plus(j, i);
}

inline FLOAT_TYPE Model::sigma2Plus(int i, int j) const {
	return (s2(j) != 0 || s2(i) != 1 || n2(i) != n2(j)) ? 0.0 : 1.0;
}

inline FLOAT_TYPE Model::sigma2Minus(int i, int j) const {
	return sigma2Plus(j, i);
}

inline COMPLEX_TYPE Model::a1PlusComplex(int i, int j) const {
	return {a1Plus(i,j),0};
}

inline COMPLEX_TYPE Model::a1Complex(int i, int j) const {
	return {a1(i,j),0};
}

inline COMPLEX_TYPE Model::a2PlusComplex(int i, int j) const {
	return {a2Plus(i,j),0};
}

inline COMPLEX_TYPE Model::a2Complex(int i, int j) const {
	return {a2(i,j),0};
}

inline COMPLEX_TYPE Model::L(int i, int j) const {
	// L.real = -kappa*(a1Plus.a1 + a2Plus.a2)
	FLOAT_TYPE summands[basisSize];
	for (int k = 0; k < basisSize; ++k) {
		summands[k] = a1Plus(i, k) * a1(k, j) + a2Plus(i, k) * a2(k, j);
	}

	FLOAT_TYPE real;
	ippsSum_f(summands, basisSize, &real);

	// L.imagine = -H
	return {-kappa * real, -H(i,j)};
}

inline FLOAT_TYPE Model::H(int i, int j) const {
	FLOAT_TYPE summands[basisSize + 1]; // plus the term after the cycle
	for (int k = 0; k < basisSize; ++k) {
		summands[k] = deltaOmega
				* (a1Plus(i, k) * a1(k, j)
						+ sigma1Plus(i, k) * sigma1Minus(k, j)
						+ a2Plus(i, k) * a2(k, j)
						+ sigma2Plus(i, k) * sigma2Minus(k, j))
				- J * (a1Plus(i, k) * a2(k, j) + a2Plus(i, k) * a1(k, j))
				+ g
						* (a1(i, k) * sigma1Plus(k, j)
								+ a1Plus(i, k) * sigma1Minus(k, j)
								+ a2(i, k) * sigma2Plus(k, j)
								+ a2Plus(i, k) * sigma2Minus(k, j));
	}

	summands[basisSize] = scE
			* (a1Plus(i, j) + a1(i, j) + a2Plus(i, j) + a2(i, j));

	FLOAT_TYPE result;
	ippsSum_f(summands, basisSize + 1, &result);

	return result;
}

inline CSR3Matrix *Model::createCSR3Matrix(CalcElemFuncP f) const {

	int totalValuesNumber = basisSize * basisSize;
	COMPLEX_TYPE *denseMatrix = createMatrix(f);

	int job[] = { //
			0, // to CSR
					0, // zero-based indexing of the dense matrix
					0, // zero-based indexing of the CSR form
					2, // zero-based indexing of the CSR form
					totalValuesNumber, // max nonzero elements
					1 // generate all
			};

	CSR3Matrix *csr3Matrix = new CSR3Matrix(basisSize, basisSize);

	int info = 1;

	complex_mkl_dnscsr(job, &basisSize, &basisSize, denseMatrix, &basisSize,
			csr3Matrix->values, csr3Matrix->columns, csr3Matrix->rowIndex,
			&info);

	return csr3Matrix;
}

inline COMPLEX_TYPE *Model::createMatrix(CalcElemFuncP f) const {
	COMPLEX_TYPE *denseMatrix = new COMPLEX_TYPE[basisSize * basisSize];
	for (int i = 0; i < basisSize; ++i) {
		for (int j = 0; j < basisSize; ++j) {
			denseMatrix[i * basisSize + j] = (this->*f)(i, j);
		}
	}

	return denseMatrix;
}
