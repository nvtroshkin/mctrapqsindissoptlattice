/*
 * utilities.h
 *
 *  Created on: Nov 24, 2017
 *      Author: fakesci
 */

#ifndef SRC_INCLUDE_UTILITIES_H_
#define SRC_INCLUDE_UTILITIES_H_

#include <iostream>
#include "precision-definition.h"

/**
 * Prints a row:
 *
 * "title: {array_elements_separated_by_commas}"
 */
inline void print(std::ostream &os, const char * title,
		const CUDA_COMPLEX_TYPE * array, const int arraySize) {
	os << title << ": {" << std::endl;
	for (int v = 0; v < arraySize; v++) {
		os << array[v].x << " + " << array[v].y << "i, ";
	}
	os << "}" << std::endl;
}

/**
 * Prints a row:
 *
 * "title: {array_elements_separated_by_commas}"
 */
template<typename T>
inline void print(std::ostream &os, const char * title,
		const T * array, const int arraySize) {
	os << title << ": {" << std::endl;
	for (int v = 0; v < arraySize; v++) {
		os << array[v] << ", ";
	}
	os << "}" << std::endl;
}

/**
 * Prints a matrix:
 *
 * "title:
 * {
 *   { v1, v2, v3, },
 *   { ... },
 * }"
 */
inline void print(std::ostream &os, const char * title, const CUDA_COMPLEX_TYPE * const * const array,
		const int width, const int height) {
	os << title << ":" << std::endl << "{" << std::endl;
	for (int i = 0; i < height; ++i) {
		os << "  { ";
		for (int j = 0; j < width; j++) {
			os << array[i][j].x << " + " << array[i][j].y << "i, ";
		}
		os << " }," << std::endl;
	}
	os << "}" << std::endl;
}

/**
 * Prints a matrix:
 *
 * "title:
 * {
 *   { v1, v2, v3, },
 *   { ... },
 * }"
 */
inline void print(std::ostream &os, const char * title, const CUDA_COMPLEX_TYPE * const array,
		const int width, const int height) {
	os << title << ":" << std::endl << "{" << std::endl;
	for (int i = 0; i < height; ++i) {
		os << "  { ";
		for (int j = 0; j < width; j++) {
			os << array[i * width + j].x << " + " << array[i * width + j].y
					<< "i, ";
		}
		os << " }," << std::endl;
	}
	os << "}" << std::endl;
}

/**
 * Prints a row to the file:
 *
 * "title: {array_elements_separated_by_commas}"
 */
//void print(const char *fileName, const FLOAT_TYPE *array, const int size) {
//	std::ofstream myfile(fileName);
//	myfile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
//	for (int i = 0; i < size; i++) {
//		myfile << array[i] << ", ";
//	}
//	myfile.close();
//}

#endif /* SRC_INCLUDE_UTILITIES_H_ */
