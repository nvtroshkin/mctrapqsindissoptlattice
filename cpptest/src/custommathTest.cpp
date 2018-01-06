/*
 * custommathTest.cpp
 *
 *  Created on: Jan 5, 2018
 *      Author: fakesci
 */

#include <functional>

#include "definitions.h"
#include "custommathTest0.h"
#include "eval-params.h"
#include "precision-definition.h"

#include "helper_cuda.h"

struct _ValueFunctor {
	virtual const char * getName() const = 0;
	virtual CUDA_COMPLEX_TYPE operator()(const uint vSize, const uint i,
			const uint j) const = 0;
	virtual ~_ValueFunctor() {
	}
	;
};

struct _vSizeMinusIvSizeMinusJ: public _ValueFunctor {
	const char * getName() const override {
		return "{vSize -i,vSize -j}";
	}

	CUDA_COMPLEX_TYPE operator()(const uint vSize, const uint i,
			const uint j) const override {
		return {vSize -i,vSize -j};
	}
};

struct _realOne: public _ValueFunctor {
	const char * getName() const override {
		return "{1,0}";
	}

	CUDA_COMPLEX_TYPE operator()(const uint vSize, const uint i,
			const uint j) const override {
		return {1,0};
	}
};

struct _imagOne: public _ValueFunctor {
	const char * getName() const override {
		return "{0,1}";
	}

	CUDA_COMPLEX_TYPE operator()(const uint vSize, const uint i,
			const uint j) const override {
		return {0,1};
	}
};

struct _realIPercent2: public _ValueFunctor {
	const char * getName() const override {
		return "{i%2,0}";
	}

	CUDA_COMPLEX_TYPE operator()(const uint vSize, const uint i,
			const uint j) const override {
		return {i%2,0};
	}
};

struct _imagIPercent2: public _ValueFunctor {
	const char * getName() const override {
		return "{0,i%2}";
	}

	CUDA_COMPLEX_TYPE operator()(const uint vSize, const uint i,
			const uint j) const override {
		return {0,i%2};
	}
};

struct _realJPercent2: public _ValueFunctor {
	const char * getName() const override {
		return "{j%2,0}";
	}

	CUDA_COMPLEX_TYPE operator()(const uint vSize, const uint i,
			const uint j) const override {
		return {j%2,0};
	}
};

struct _imagJPercent2: public _ValueFunctor {
	const char * getName() const override {
		return "{0,j%2}";
	}

	CUDA_COMPLEX_TYPE operator()(const uint vSize, const uint i,
			const uint j) const override {
		return {0,j%2};
	}
};

struct _realIPercent2MultJPercent2: public _ValueFunctor {
	const char * getName() const override {
		return "{i%2 * j%2,0}";
	}

	CUDA_COMPLEX_TYPE operator()(const uint vSize, const uint i,
			const uint j) const override {
		return {i%2 * j%2,0};
	}
};

struct _imagIPercent2MultJPercent2: public _ValueFunctor {
	const char * getName() const override {
		return "{0,i%2 * j%2}";
	}

	CUDA_COMPLEX_TYPE operator()(const uint vSize, const uint i,
			const uint j) const override {
		return {0,i%2 * j%2};
	}
};

void createCSR3Matrix(cuFloatComplex * matrix, uint mSize, uint vSize,
		cuFloatComplex * values, int * columns, int * rowIndex) {

	int job[] = { //
			0, // to CSR
					0, // zero-based indexing of the dense matrix
					0, // zero-based indexing of the CSR form
					2, // zero-based indexing of the CSR form
					mSize, // max nonzero elements
					1 // generate all
			};

	int info = 1;

	int basisSizeInt = vSize;

	complex_mkl_dnscsr(job, &basisSizeInt, &basisSizeInt,
			(COMPLEX_TYPE *) matrix, &basisSizeInt, (COMPLEX_TYPE *) values,
			columns, rowIndex, &info);
}

void createMatrix(const int vSize, CUDA_COMPLEX_TYPE * const matrix,
		const _ValueFunctor &getValue) {
	for (int i = 0; i < vSize; ++i) {
		for (int j = 0; j < vSize; ++j) {
			matrix[i * vSize + j] = getValue(vSize, i, j);
		}
	}
}

void createVector(const int vSize, CUDA_COMPLEX_TYPE * const vector,
		const _ValueFunctor &getValue) {
	for (int i = 0; i < vSize; ++i) {
		vector[i] = getValue(vSize, i, i);
	}
}

void calcResult(const uint vSize, const uint nRows,
		const CUDA_COMPLEX_TYPE * const matrix,
		const CUDA_COMPLEX_TYPE * const vector,
		CUDA_COMPLEX_TYPE * const result) {
//Kahan's summation algorithm (https://en.wikipedia.org/wiki/Kahan_summation_algorithm)
	for (uint i = 0; i < nRows; ++i) {
		CUDA_COMPLEX_TYPE r = { 0, 0 };
		CUDA_COMPLEX_TYPE c = { 0, 0 };
		for (uint j = 0; j < vSize; ++j) {
			CUDA_COMPLEX_TYPE y = { matrix[i * vSize + j].x * vector[j].x
					- matrix[i * vSize + j].y * vector[j].y - c.x, matrix[i
					* vSize + j].x * vector[j].y
					+ matrix[i * vSize + j].y * vector[j].x - c.y };

			CUDA_COMPLEX_TYPE t = { r.x + y.x, r.y + y.y };

			c.x = (t.x - r.x) - y.x;
			c.y = (t.y - r.y) - y.y;

			r.x = t.x;
			r.y = t.y;
		}

		result[i].x = r.x;
		result[i].y = r.y;
	}
}

void calcResult(const uint vSize, const uint * const checkRows,
		const uint checkRowsSize, const CUDA_COMPLEX_TYPE * const matrix,
		const CUDA_COMPLEX_TYPE * const vector,
		CUDA_COMPLEX_TYPE * const result) {
	//Kahan's summation algorithm (https://en.wikipedia.org/wiki/Kahan_summation_algorithm)
	for (uint i = 0; i < checkRowsSize; ++i) {
		uint rowBegin = checkRows[i] * vSize;

		CUDA_COMPLEX_TYPE r = { 0, 0 };
		CUDA_COMPLEX_TYPE c = { 0, 0 };
		for (uint j = 0; j < vSize; ++j) {
			CUDA_COMPLEX_TYPE y = { matrix[rowBegin + j].x * vector[j].x
					- matrix[rowBegin + j].y * vector[j].y - c.x,
					matrix[rowBegin + j].x * vector[j].y
							+ matrix[rowBegin + j].y * vector[j].x - c.y };

			CUDA_COMPLEX_TYPE t = { r.x + y.x, r.y + y.y };

			c.x = (t.x - r.x) - y.x;
			c.y = (t.y - r.y) - y.y;

			r.x = t.x;
			r.y = t.y;
		}

		result[i].x = r.x;
		result[i].y = r.y;
	}
}

std::string getCaseId(const uint size, const uint nWarpsPerBloc,
		const uint ilpColumn, const uint ilpRow,
		const _ValueFunctor &getMatrixValue,
		const _ValueFunctor &getVectorValue) {
	return std::string(
			"size=" + std::to_string(size) + ", nWarps="
					+ std::to_string(nWarpsPerBloc) + ". ilpColumn="
					+ std::to_string(ilpColumn) + ", ilpRow="
					+ std::to_string(ilpRow) + ", matrixValue="
					+ getMatrixValue.getName() + ", vectorValue="
					+ getVectorValue.getName());
}

template<uint size, uint nWarpsPerBlock, uint ilpColumn>
void checkVectorVectorTestCase(const _ValueFunctor &getVectorValue1,
		const _ValueFunctor &getVectorValue2) {

	const uint vSize = size;

	CUDA_COMPLEX_TYPE * v1 = new CUDA_COMPLEX_TYPE[vSize];
	createVector(vSize, v1, getVectorValue1);

	CUDA_COMPLEX_TYPE * v2 = new CUDA_COMPLEX_TYPE[vSize];
	createVector(vSize, v2, getVectorValue2);

	size_t vSizet = vSize * sizeof(CUDA_COMPLEX_TYPE);

	CUDA_COMPLEX_TYPE * v1DevPtr, *v2DevPtr, *actualResultDevPtr;
	checkCudaErrors(cudaMalloc((void ** ) &v1DevPtr, vSizet));
	checkCudaErrors(cudaMalloc((void ** ) &v2DevPtr, vSizet));
	checkCudaErrors(cudaMalloc((void ** ) &actualResultDevPtr, vSizet));

	checkCudaErrors(cudaMemcpy(v1DevPtr, v1, vSizet, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(v2DevPtr, v2, vSizet, cudaMemcpyHostToDevice));

	CUDA_COMPLEX_TYPE expectedResult[1];
	calcResult(vSize, 1, v1, v2, expectedResult);

	testMultVectorVector<vSize, nWarpsPerBlock * CUDA_WARP_SIZE, ilpColumn>(v1DevPtr,
			v2DevPtr, actualResultDevPtr);

	_checkDeviceState(
			getCaseId(size, nWarpsPerBlock, ilpColumn, 1, getVectorValue1,
					getVectorValue2), 1, actualResultDevPtr, expectedResult,
			RIGHT_DIGITS);

	cudaDeviceReset();

	delete[] v1;
	delete[] v2;
}

TEST (custommath, multVectorVector) {
	//check simple cases
	checkVectorVectorTestCase<8 * 1024, 2, 1>(_realOne { }, _realOne { });
	checkVectorVectorTestCase<8 * 1024, 2, 1>(_realOne { }, _imagOne { });
	checkVectorVectorTestCase<8 * 1024, 2, 1>(_imagOne { }, _realOne { });
	checkVectorVectorTestCase<8 * 1024, 2, 1>(_imagOne { }, _imagOne { });

	checkVectorVectorTestCase<8 * 1024, 2, 1>(_realIPercent2 { }, _realOne { });
	checkVectorVectorTestCase<8 * 1024, 2, 1>(_imagIPercent2 { }, _realOne { });
	checkVectorVectorTestCase<8 * 1024, 2, 1>(_realOne { }, _realIPercent2 { });
	checkVectorVectorTestCase<8 * 1024, 2, 1>(_realOne { }, _imagIPercent2 { });

	checkVectorVectorTestCase<8 * 1024, 2, 1>(_realJPercent2 { }, _realOne { });
	checkVectorVectorTestCase<8 * 1024, 2, 1>(_imagJPercent2 { }, _realOne { });
	checkVectorVectorTestCase<8 * 1024, 2, 1>(_realOne { }, _realJPercent2 { });
	checkVectorVectorTestCase<8 * 1024, 2, 1>(_realOne { }, _imagJPercent2 { });

	checkVectorVectorTestCase<8 * 1024, 2, 1>(_realIPercent2MultJPercent2 { },
			_realOne { });
	checkVectorVectorTestCase<8 * 1024, 2, 1>(_imagIPercent2MultJPercent2 { },
			_realOne { });
	checkVectorVectorTestCase<8 * 1024, 2, 1>(_realOne { },
			_realIPercent2MultJPercent2 { });
	checkVectorVectorTestCase<8 * 1024, 2, 1>(_realOne { },
			_imagIPercent2MultJPercent2 { });

	//check ilp
	checkVectorVectorTestCase<8 * 1024, 2, 1>(_vSizeMinusIvSizeMinusJ { },
			_vSizeMinusIvSizeMinusJ { });
	checkVectorVectorTestCase<8 * 1024, 2, 2>(_vSizeMinusIvSizeMinusJ { },
			_vSizeMinusIvSizeMinusJ { });

	//check remainings
	checkVectorVectorTestCase<8 * 1024, 3, 1>(_vSizeMinusIvSizeMinusJ { },
			_vSizeMinusIvSizeMinusJ { });
	checkVectorVectorTestCase<8 * 1024, 2, 3>(_vSizeMinusIvSizeMinusJ { },
			_vSizeMinusIvSizeMinusJ { });
}

template<uint size, uint nWarpsPerBlock, uint ilpColumn, uint ilpRow>
void checkMatrixVectorTestCase(const _ValueFunctor &getMatrixValue,
		const _ValueFunctor &getVectorValue) {

	const uint vSize = size;
	const uint mSize = vSize * vSize;

	CUDA_COMPLEX_TYPE * vector = new CUDA_COMPLEX_TYPE[vSize];
	createVector(vSize, vector, getVectorValue);

	CUDA_COMPLEX_TYPE * matrix = new CUDA_COMPLEX_TYPE[mSize];
	createMatrix(vSize, matrix, getMatrixValue);

	size_t mSizet = mSize * sizeof(CUDA_COMPLEX_TYPE);
	size_t vSizet = vSize * sizeof(CUDA_COMPLEX_TYPE);

	CUDA_COMPLEX_TYPE * matrixDevPtr, *vectorDevPtr, *actualResultDevPtr;
	checkCudaErrors(cudaMalloc((void ** ) &matrixDevPtr, mSizet));
	checkCudaErrors(cudaMalloc((void ** ) &vectorDevPtr, vSizet));
	checkCudaErrors(cudaMalloc((void ** ) &actualResultDevPtr, vSizet));

	checkCudaErrors(
			cudaMemcpy(matrixDevPtr, matrix, mSizet, cudaMemcpyHostToDevice));
	checkCudaErrors(
			cudaMemcpy(vectorDevPtr, vector, vSizet, cudaMemcpyHostToDevice));

	const uint checkRowsSize = 10;
	const uint checkRows[] = { 0, 1, 2, 3, 4, 5, 6, vSize / 4, vSize / 2, 3
			* vSize / 4, vSize - 1 };

	CUDA_COMPLEX_TYPE expectedResult[vSize];
	calcResult(vSize, checkRows, checkRowsSize, matrix, vector, expectedResult);

	testMultMatrixVector<vSize, CUDA_WARP_SIZE * nWarpsPerBlock, ilpColumn, ilpRow>(
			matrixDevPtr, vectorDevPtr, actualResultDevPtr);

	CUDA_COMPLEX_TYPE * actualResult = new CUDA_COMPLEX_TYPE[vSize];
	checkCudaErrors(
			cudaMemcpy(actualResult, actualResultDevPtr, vSizet,
					cudaMemcpyDeviceToHost));

	CUDA_COMPLEX_TYPE actualResultCheckElemets[checkRowsSize];
	for (uint i = 0; i < checkRowsSize; ++i) {
		actualResultCheckElemets[i] = actualResult[checkRows[i]];
	}

	_checkState(
			getCaseId(size, nWarpsPerBlock, ilpColumn, ilpRow, getMatrixValue,
					getVectorValue), checkRowsSize, actualResultCheckElemets,
			expectedResult, RIGHT_DIGITS);

	cudaDeviceReset();

	delete[] vector;
	delete[] matrix;
	delete[] actualResult;
}

TEST (custommath, multMatrixVector) {
	//check simple cases
	checkMatrixVectorTestCase<1024, 2, 1, 1>(_realOne { }, _realOne { });
	checkMatrixVectorTestCase<1024, 2, 1, 1>(_realOne { }, _imagOne { });
	checkMatrixVectorTestCase<1024, 2, 1, 1>(_imagOne { }, _realOne { });
	checkMatrixVectorTestCase<1024, 2, 1, 1>(_imagOne { }, _imagOne { });

	checkMatrixVectorTestCase<1024, 2, 1, 1>(_realIPercent2 { }, _realOne { });
	checkMatrixVectorTestCase<1024, 2, 1, 1>(_imagIPercent2 { }, _realOne { });
	checkMatrixVectorTestCase<1024, 2, 1, 1>(_realOne { }, _realIPercent2 { });
	checkMatrixVectorTestCase<1024, 2, 1, 1>(_realOne { }, _imagIPercent2 { });

	checkMatrixVectorTestCase<1024, 2, 1, 1>(_realJPercent2 { }, _realOne { });
	checkMatrixVectorTestCase<1024, 2, 1, 1>(_imagJPercent2 { }, _realOne { });
	checkMatrixVectorTestCase<1024, 2, 1, 1>(_realOne { }, _realJPercent2 { });
	checkMatrixVectorTestCase<1024, 2, 1, 1>(_realOne { }, _imagJPercent2 { });

	checkMatrixVectorTestCase<1024, 2, 1, 1>(_realIPercent2MultJPercent2 { },
			_realOne { });
	checkMatrixVectorTestCase<1024, 2, 1, 1>(_imagIPercent2MultJPercent2 { },
			_realOne { });
	checkMatrixVectorTestCase<1024, 2, 1, 1>(_realOne { },
			_realIPercent2MultJPercent2 { });
	checkMatrixVectorTestCase<1024, 2, 1, 1>(_realOne { },
			_imagIPercent2MultJPercent2 { });

	//check ilp
	checkMatrixVectorTestCase<8 * 1024, 8, 1, 1>(_vSizeMinusIvSizeMinusJ { },
			_vSizeMinusIvSizeMinusJ { });
	checkMatrixVectorTestCase<8 * 1024, 8, 2, 1>(_vSizeMinusIvSizeMinusJ { },
			_vSizeMinusIvSizeMinusJ { });
	checkMatrixVectorTestCase<8 * 1024, 8, 1, 2>(_vSizeMinusIvSizeMinusJ { },
			_vSizeMinusIvSizeMinusJ { });

	//check remainings
	checkMatrixVectorTestCase<8 * 1024, 12, 1, 1>(_vSizeMinusIvSizeMinusJ { },
			_vSizeMinusIvSizeMinusJ { });
	checkMatrixVectorTestCase<8 * 1024, 8, 3, 1>(_vSizeMinusIvSizeMinusJ { },
			_vSizeMinusIvSizeMinusJ { });
	checkMatrixVectorTestCase<8 * 1024, 8, 1, 3>(_vSizeMinusIvSizeMinusJ { },
			_vSizeMinusIvSizeMinusJ { });
	checkMatrixVectorTestCase<8 * 1024, 4, 3, 3>(_vSizeMinusIvSizeMinusJ { },
			_vSizeMinusIvSizeMinusJ { });
}

template<uint size, uint nWarpsPerBlock, uint ilpColumn>
void checkSparseMatrixVectorTestCase(const _ValueFunctor &getMatrixValue,
		const _ValueFunctor &getVectorValue) {

	const uint vSize = size;
	const uint mSize = vSize * vSize;

	CUDA_COMPLEX_TYPE * vector = new CUDA_COMPLEX_TYPE[vSize];
	createVector(vSize, vector, getVectorValue);

	CUDA_COMPLEX_TYPE * matrix = new CUDA_COMPLEX_TYPE[mSize];
	createMatrix(vSize, matrix, getMatrixValue);

	cuFloatComplex * values = new cuFloatComplex[mSize];
	int * columns = new int[mSize];
	int * rowIndex = new int[vSize + 1];
	createCSR3Matrix(matrix, mSize, vSize, values, columns, rowIndex);

	size_t mSizet = mSize * sizeof(cuFloatComplex);
	size_t vSizet = vSize * sizeof(cuFloatComplex);

	size_t columnsSizet = mSize * sizeof(int);
	size_t rowIndexSizet = (vSize + 1) * sizeof(int);

	cuFloatComplex * valuesDevPtr, *vectorDevPtr, *actualResultDevPtr;
	int * columnsDevPtr, *rowIndexDevPtr;

	checkCudaErrors(cudaMalloc((void ** ) &valuesDevPtr, mSizet));
	checkCudaErrors(cudaMalloc((void ** ) &vectorDevPtr, vSizet));
	checkCudaErrors(cudaMalloc((void ** ) &actualResultDevPtr, vSizet));

	checkCudaErrors(cudaMalloc((void ** ) &columnsDevPtr, columnsSizet));
	checkCudaErrors(cudaMalloc((void ** ) &rowIndexDevPtr, rowIndexSizet));

	checkCudaErrors(
			cudaMemcpy(valuesDevPtr, values, mSizet, cudaMemcpyHostToDevice));
	checkCudaErrors(
			cudaMemcpy(columnsDevPtr, columns, columnsSizet,
					cudaMemcpyHostToDevice));
	checkCudaErrors(
			cudaMemcpy(rowIndexDevPtr, rowIndex, rowIndexSizet,
					cudaMemcpyHostToDevice));
	checkCudaErrors(
			cudaMemcpy(vectorDevPtr, vector, vSizet, cudaMemcpyHostToDevice));

	const uint checkRowsSize = 10;
	const uint checkRows[] = { 0, 1, 2, 3, 4, 5, 6, vSize / 4, vSize / 2, 3
			* vSize / 4, vSize - 1 };

	CUDA_COMPLEX_TYPE expectedResult[vSize];
	calcResult(vSize, checkRows, checkRowsSize, matrix, vector, expectedResult);

	testMultSparseMatrixVector<vSize, CUDA_WARP_SIZE * nWarpsPerBlock, ilpColumn>(
			valuesDevPtr, columnsDevPtr, rowIndexDevPtr, vectorDevPtr, actualResultDevPtr);

	CUDA_COMPLEX_TYPE * actualResult = new CUDA_COMPLEX_TYPE[vSize];
	checkCudaErrors(
			cudaMemcpy(actualResult, actualResultDevPtr, vSizet,
					cudaMemcpyDeviceToHost));

	CUDA_COMPLEX_TYPE actualResultCheckElemets[checkRowsSize];
	for (uint i = 0; i < checkRowsSize; ++i) {
		actualResultCheckElemets[i] = actualResult[checkRows[i]];
	}

	_checkState(
			getCaseId(size, nWarpsPerBlock, ilpColumn, 1, getMatrixValue,
					getVectorValue), checkRowsSize, actualResultCheckElemets,
			expectedResult, RIGHT_DIGITS);

	cudaDeviceReset();

	delete[] vector;
	delete[] matrix;
	delete[] actualResult;
	delete[] values;
	delete[] columns;
	delete[] rowIndex;
}

TEST (custommath, multSparseMatrixVector) {
	//check simple cases
	checkSparseMatrixVectorTestCase<1024, 2, 1>(_realOne { }, _realOne { });
	checkSparseMatrixVectorTestCase<1024, 2, 1>(_realOne { }, _imagOne { });
	checkSparseMatrixVectorTestCase<1024, 2, 1>(_imagOne { }, _realOne { });
	checkSparseMatrixVectorTestCase<1024, 2, 1>(_imagOne { }, _imagOne { });

	checkSparseMatrixVectorTestCase<1024, 2, 1>(_realIPercent2 { }, _realOne { });
	checkSparseMatrixVectorTestCase<1024, 2, 1>(_imagIPercent2 { }, _realOne { });
	checkSparseMatrixVectorTestCase<1024, 2, 1>(_realOne { }, _realIPercent2 { });
	checkSparseMatrixVectorTestCase<1024, 2, 1>(_realOne { }, _imagIPercent2 { });

	checkSparseMatrixVectorTestCase<1024, 2, 1>(_realJPercent2 { }, _realOne { });
	checkSparseMatrixVectorTestCase<1024, 2, 1>(_imagJPercent2 { }, _realOne { });
	checkSparseMatrixVectorTestCase<1024, 2, 1>(_realOne { }, _realJPercent2 { });
	checkSparseMatrixVectorTestCase<1024, 2, 1>(_realOne { }, _imagJPercent2 { });

	checkSparseMatrixVectorTestCase<1024, 2, 1>(_realIPercent2MultJPercent2 { },
			_realOne { });
	checkSparseMatrixVectorTestCase<1024, 2, 1>(_imagIPercent2MultJPercent2 { },
			_realOne { });
	checkSparseMatrixVectorTestCase<1024, 2, 1>(_realOne { },
			_realIPercent2MultJPercent2 { });
	checkSparseMatrixVectorTestCase<1024, 2, 1>(_realOne { },
			_imagIPercent2MultJPercent2 { });

	//check ilp
	checkSparseMatrixVectorTestCase<8 * 1024, 8, 1>(_vSizeMinusIvSizeMinusJ { },
			_vSizeMinusIvSizeMinusJ { });
	checkSparseMatrixVectorTestCase<8 * 1024, 8, 2>(_vSizeMinusIvSizeMinusJ { },
			_vSizeMinusIvSizeMinusJ { });

	//check remainings
	checkSparseMatrixVectorTestCase<8 * 1024, 12, 1>(_vSizeMinusIvSizeMinusJ { },
			_vSizeMinusIvSizeMinusJ { });
	checkSparseMatrixVectorTestCase<8 * 1024, 8, 3>(_vSizeMinusIvSizeMinusJ { },
			_vSizeMinusIvSizeMinusJ { });
}

