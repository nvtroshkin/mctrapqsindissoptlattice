/*
 * CSR3Matrix.cpp
 *
 *  Created on: Nov 29, 2017
 *      Author: fakesci
 */

#include <precision-definition.h>
#include <CSR3Matrix.h>

CSR3Matrix::CSR3Matrix(int rowsNumber, int nonZeroValuesNumber) :
		rowsNumber(rowsNumber), nonZeroValuesNumber(nonZeroValuesNumber), values(
				new COMPLEX_TYPE[nonZeroValuesNumber]), columns(
				new int[nonZeroValuesNumber]), rowIndex(
				new int[rowsNumber + 1]/*non-zero element on each row*/) {
	//put the length of the values array at the end
	rowIndex[rowsNumber] = nonZeroValuesNumber;
}

CSR3Matrix::~CSR3Matrix() {
	delete[] values;
	delete[] columns;
	delete[] rowIndex;
}
