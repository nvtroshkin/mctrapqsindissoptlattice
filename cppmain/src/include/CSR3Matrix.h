/*
 * CSR3Matrix.h
 *
 *  Created on: Nov 29, 2017
 *      Author: fakesci
 */

#ifndef SRC_INCLUDE_CSR3MATRIX_H_
#define SRC_INCLUDE_CSR3MATRIX_H_

#include <precision-definition.h>

/**
 * The CSR3 Storage Format, the three array variation
 *
 * See: https://software.intel.com/en-us/mkl-developer-reference-c-sparse-blas-csr-matrix-storage-format
 */
struct CSR3Matrix {

	/**
	 * The total number of rows
	 */
	int rowsNumber;

	/**
	 * The non-zero elements
	 */
	CUDA_COMPLEX_TYPE *values;

	/**
	 * The i-th element value is the column of the i-th value in values
	 */
	int *columns;

	/**
	 * The i-th element is the index of the element in values
	 * that is the first non-zero element in the i-th row of the matrix
	 *
	 * the last element - the total number of elements in values (equals rowsNumber)
	 */
	int *rowIndex;

	CSR3Matrix(int rowsNumber, int columnsNumber);
	~CSR3Matrix();
};



#endif /* SRC_INCLUDE_CSR3MATRIX_H_ */
