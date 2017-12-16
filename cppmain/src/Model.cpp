/*
 *  Created on: Nov 18, 2017
 *      Author: fakesci
 */

#include <Model.h>
#include <utilities.h>
#include <mkl-constants.h>

Model::Model(MKL_INT atom1SSize, MKL_INT atom2SSize, MKL_INT atom3SSize,
MKL_INT field1SSize,
MKL_INT field2SSize, MKL_INT field3SSize,
FLOAT_TYPE kappa, FLOAT_TYPE deltaOmega, FLOAT_TYPE g,
FLOAT_TYPE scE, FLOAT_TYPE J) :
		atom1SSize(atom1SSize), field1SSize(field1SSize), subs1Size(
				atom1SSize * field1SSize), atom2SSize(atom2SSize), field2SSize(
				field2SSize), subs2Size(atom2SSize * field2SSize), atom3SSize(
				atom3SSize), field3SSize(field3SSize), subs3Size(
				atom3SSize * field3SSize), basisSize(
				subs1Size * subs2Size * subs3Size), kappa(kappa), deltaOmega(
				deltaOmega), g(g), scE(scE), J(J) {
	MKL_INT maxFieldSSize = std::max(std::max(field1SSize, field2SSize),
			field3SSize);

	sqrtsOfPhotonNumbers = new FLOAT_TYPE[maxFieldSSize];
	FLOAT_TYPE photonNumbers[maxFieldSSize];
	for (int k = 0; k < maxFieldSSize; ++k) {
		photonNumbers[k] = k;
	}
	vSqrt(maxFieldSSize, photonNumbers, sqrtsOfPhotonNumbers);

	a1InCSR3 = createCSR3Matrix(&Model::a1Complex, "A1");
	a1PlusInCSR3 = createCSR3Matrix(&Model::a1PlusComplex, "A1+");

	a2InCSR3 = createCSR3Matrix(&Model::a2Complex, "A2");
	a2PlusInCSR3 = createCSR3Matrix(&Model::a2PlusComplex, "A2Plus");

	a3InCSR3 = createCSR3Matrix(&Model::a3Complex, "A3");
	a3PlusInCSR3 = createCSR3Matrix(&Model::a3PlusComplex, "A3Plus");
#ifdef L_SPARSE
	lInCSR3 = createCSR3Matrix(&Model::L, "L");
#else
	l = createMatrix(&Model::L, "L");
#endif
}

Model::~Model() {
	delete[] sqrtsOfPhotonNumbers;
	delete a1InCSR3;
	delete a1PlusInCSR3;
	delete a2InCSR3;
	delete a2PlusInCSR3;
	delete a3InCSR3;
	delete a3PlusInCSR3;

#ifdef L_SPARSE
	delete lInCSR3;
#else
	delete[] l;
#endif
}

//------------------------ The first cavity ------------------------------

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

inline COMPLEX_TYPE Model::a1PlusComplex(int i, int j) const {
	return {a1Plus(i,j),0};
}

inline COMPLEX_TYPE Model::a1Complex(int i, int j) const {
	return {a1(i,j),0};
}

//------------------------ The second cavity ------------------------------

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

inline COMPLEX_TYPE Model::a2PlusComplex(int i, int j) const {
	return {a2Plus(i,j),0};
}

inline COMPLEX_TYPE Model::a2Complex(int i, int j) const {
	return {a2(i,j),0};
}

//------------------------ The third cavity ------------------------------

inline FLOAT_TYPE Model::a3Plus(int i, int j) const {
	return (n3(i) != n3(j) + 1 || s3(i) != s3(j)) ?
			0.0 : sqrtsOfPhotonNumbers[n3(j) + 1];
}

inline FLOAT_TYPE Model::a3(int i, int j) const {
	return a3Plus(j, i);
}

inline FLOAT_TYPE Model::sigma3Plus(int i, int j) const {
	return (s3(j) != 0 || s3(i) != 1 || n3(i) != n3(j)) ? 0.0 : 1.0;
}

inline FLOAT_TYPE Model::sigma3Minus(int i, int j) const {
	return sigma3Plus(j, i);
}

inline COMPLEX_TYPE Model::a3PlusComplex(int i, int j) const {
	return {a3Plus(i,j),0};
}

inline COMPLEX_TYPE Model::a3Complex(int i, int j) const {
	return {a3(i,j),0};
}

//----------------------------------------------------------

inline COMPLEX_TYPE Model::L(int i, int j) const {
	// L.real = -kappa*(a1Plus.a1 + a2Plus.a2 + a3Plus.a3)
	FLOAT_TYPE summands[basisSize];
	for (int k = 0; k < basisSize; ++k) {
		summands[k] = a1Plus(i, k) * a1(k, j) + a2Plus(i, k) * a2(k, j)
				+ a3Plus(i, k) * a3(k, j);
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
				* (a1Plus(i, k) * a1(k, j) + a2Plus(i, k) * a2(k, j)
						+ a3Plus(i, k) * a3(k, j)
						+ sigma1Plus(i, k) * sigma1Minus(k, j)
						+ sigma2Plus(i, k) * sigma2Minus(k, j)
						+ sigma3Plus(i, k) * sigma3Minus(k, j))
				- J
						* (a2Plus(i, k) * (a1(k, j) + a3(k, j))
								+ (a1Plus(i, k) + a3Plus(i, k)) * a2(k, j))
				+ g
						* (a1(i, k) * sigma1Plus(k, j)
								+ a1Plus(i, k) * sigma1Minus(k, j)
								+ a2(i, k) * sigma2Plus(k, j)
								+ a2Plus(i, k) * sigma2Minus(k, j)
								+ a3(i, k) * sigma3Plus(k, j)
								+ a3Plus(i, k) * sigma3Minus(k, j));
	}

	summands[basisSize] = scE
			* (a1Plus(i, j) + a1(i, j) + a2Plus(i, j) + a2(i, j) + a3Plus(i, j)
					+ a3(i, j));

	FLOAT_TYPE result;
	ippsSum_f(summands, basisSize + 1, &result);

	return result;
}

inline CSR3Matrix *Model::createCSR3Matrix(CalcElemFuncP f,
		std::string matrixName) const {

	int totalValuesNumber = basisSize * basisSize;
	COMPLEX_TYPE *denseMatrix = createMatrix(f, matrixName);

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

	delete[] denseMatrix;

	return csr3Matrix;
}

inline COMPLEX_TYPE *Model::createMatrix(CalcElemFuncP f,
		std::string matrixName) const {
#ifdef CHECK_SPARSITY
	int nonZeroCounter = 0;
#endif

	int totalElementsCount = basisSize * basisSize;
	COMPLEX_TYPE *denseMatrix = new COMPLEX_TYPE[totalElementsCount];
	for (int i = 0; i < basisSize; ++i) {
		for (int j = 0; j < basisSize; ++j) {
			denseMatrix[i * basisSize + j] = (this->*f)(i, j);
#ifdef CHECK_SPARSITY
			if (denseMatrix[i * basisSize + j].real != 0.0
					|| denseMatrix[i * basisSize + j].imag != 0.0) {
				++nonZeroCounter;
			}
#endif
		}
	}

#ifdef CHECK_SPARSITY
	std::cout << "DensityOf(" << matrixName << ") = "
			<< 1.0 * nonZeroCounter / totalElementsCount << std::endl;
#endif

	return denseMatrix;
}
