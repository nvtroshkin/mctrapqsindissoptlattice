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
void print(std::ostream &os, const char title[], const COMPLEX_TYPE array[],
		int arraySize);

/**
 * Prints a matrix:
 *
 * "title:
 * {
 *   { v1, v2, v3, },
 *   { ... },
 * }"
 */
void print(std::ostream &os, const char title[], const COMPLEX_TYPE **array,
		int width, int height);

/**
 * Prints a matrix:
 *
 * "title:
 * {
 *   { v1, v2, v3, },
 *   { ... },
 * }"
 */
void print(std::ostream &os, const char title[], const COMPLEX_TYPE *array,
		int width, int height);

#endif /* SRC_INCLUDE_UTILITIES_H_ */
