/*
 * utilities.cpp
 *
 *  Created on: Nov 24, 2017
 *      Author: fakesci
 */

#include <precision-definition.h>
#include <iostream>
#include <fstream>

using std::endl;

void print(std::ostream &os, const char title[], const COMPLEX_TYPE array[],
		int arraySize) {
	os << title << ": {" << endl;
	for (int v = 0; v < arraySize; v++) {
		os << array[v].real << " + " << array[v].imag << "i, ";
	}
	os << "}" << endl;
}

void print(std::ostream &os, const char title[], COMPLEX_TYPE **array,
		int width, int height) {
	os << title << ":" << endl << "{" << endl;
	for (int i = 0; i < height; ++i) {
		os << "  { ";
		for (int j = 0; j < width; j++) {
			os << array[i][j].real << " + " << array[i][j].imag << "i, ";
		}
		os << " }," << endl;
	}
	os << "}" << endl;
}

void print(std::ostream &os, const char title[], const COMPLEX_TYPE *array,
		int width, int height) {
	os << title << ":" << endl << "{" << endl;
	for (int i = 0; i < height; ++i) {
		os << "  { ";
		for (int j = 0; j < width; j++) {
			os << array[i * width + j].real << " + "
					<< array[i * width + j].imag << "i, ";
		}
		os << " }," << endl;
	}
	os << "}" << endl;
}

void print(const char *fileName, const FLOAT_TYPE *array, const int size) {
	std::ofstream myfile(fileName);
	myfile.exceptions ( std::ifstream::failbit | std::ifstream::badbit );
	for (int i = 0; i < size; i++) {
		myfile << array[i] << ", ";
	}
	myfile.close();
}

