/*
 * CSR3Matrix.h
 *
 *  Created on: Nov 29, 2017
 *      Author: fakesci
 */

#ifndef SRC_INCLUDE_CSR3MATRIX_H_
#define SRC_INCLUDE_CSR3MATRIX_H_

#include <precision-definition.h>

struct CSR3Matrix {

	int rowsNumber;
	int nonZeroValuesNumber;
	//
	COMPLEX_TYPE *values;
	int *columns;
	int *rowIndex;	//indices from values of the first
	//non-null row elements of the matrix being compressed
	//the last element - total number of elements in values

	CSR3Matrix(int rowsNumber, int nonZeroValuesNumber);
	~CSR3Matrix();
};



#endif /* SRC_INCLUDE_CSR3MATRIX_H_ */
