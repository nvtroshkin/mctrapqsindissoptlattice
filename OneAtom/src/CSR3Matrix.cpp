/*
 * CSR3Matrix.cpp
 *
 *  Created on: Nov 29, 2017
 *      Author: fakesci
 */

#include <precision-definition.h>
#include <CSR3Matrix.h>

CSR3Matrix::CSR3Matrix(int rowsNumber, int columnsNumber) :
		rowsNumber(rowsNumber), values(
				new COMPLEX_TYPE[rowsNumber * columnsNumber]), columns(
				new int[rowsNumber * columnsNumber]), rowIndex(
				new int[rowsNumber + 1]) {
}

CSR3Matrix::~CSR3Matrix() {
	delete[] values;
	delete[] columns;
	delete[] rowIndex;
}
