/*
 * utilities.cpp
 *
 *  Created on: Nov 24, 2017
 *      Author: fake_sci
 */

#include <precision-definition.h>
#include <iostream>

using namespace std;

void print(char title[], COMPLEX_TYPE array[], int arraySize) {
	cout << title << ": {" << endl;
	for (int v = 0; v < arraySize; v++) {
		cout << array[v].real << " + " << array[v].imag << "i, ";
	}
	cout << "}" << endl;
}

