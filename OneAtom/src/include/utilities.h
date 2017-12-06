/*
 * utilities.h
 *
 *  Created on: Nov 24, 2017
 *      Author: fakesci
 */

#ifndef SRC_INCLUDE_UTILITIES_H_
#define SRC_INCLUDE_UTILITIES_H_

#include <precision-definition.h>
#include <iostream>

/**
 * Prints a row:
 *
 * "title: {array_elements_separated_by_commas}"
 */
void print(std::ostream &os, const char title[], const COMPLEX_TYPE array[],
		int arraySize);

#endif /* SRC_INCLUDE_UTILITIES_H_ */