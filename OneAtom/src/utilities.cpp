/*
 * utilities.cpp
 *
 *  Created on: Nov 24, 2017
 *      Author: fakesci
 */

#include <precision-definition.h>
#include <iostream>

using std::endl;

void print(std::ostream &os, const char title[], const COMPLEX_TYPE array[], int arraySize) {
	os << title << ": {" << endl;
	for (int v = 0; v < arraySize; v++) {
		os << array[v].real << " + " << array[v].imag << "i, ";
	}
	os << "}" << endl;
}

