/*
 * misc.h
 *
 *  Created on: Dec 8, 2017
 *      Author: fakesci
 */

#ifndef SRC_DEFINITIONS_H_
#define SRC_DEFINITIONS_H_

#include <RndNumProvider.h>
#include <cmath>
#include <iostream>
#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "precision-definition.h"
#include <string>

using namespace testing;

#define ASSERT_COMPLEX(c, expReal, expImagine)\
{											\
	ASSERT_FLOAT_EQ(c.real, expReal);\
	ASSERT_FLOAT_EQ(c.imag, expImagine);\
}

MATCHER_P (ComplexNumberEquals, a, ""){
//	*result_listener << "where the remainder is " << (arg % n);
return a.real == arg.real && a.imag == arg.imag;
}

int getDecimalExponent(FLOAT_TYPE f) {
	int exp = 0;
	if (f > 1.0) {
		while (f >= 10) {
			f /= 10;
			exp++;
		}
	} else if (f < 1.0) {
		while (f < 1.0) {
			f *= 10;
			exp--;
		}
	}

	return exp;
}

bool areEqualToNDigits(FLOAT_TYPE a, FLOAT_TYPE b, int nDigits) {
	if (a == b) {
		return true;
	}

	FLOAT_TYPE diff = std::abs(a - b);
	FLOAT_TYPE max = std::max(std::abs(a), std::abs(b));

	int diffExp = getDecimalExponent(diff), maxExp = getDecimalExponent(max);

	return diffExp <= maxExp && std::abs(maxExp - diffExp) >= nDigits;
}

bool areEqualToNDigits(COMPLEX_TYPE a, COMPLEX_TYPE b, int nDigits) {
	return areEqualToNDigits(a.real, b.real, nDigits)
			&& areEqualToNDigits(a.imag, b.imag, nDigits);
}

MATCHER_P (ComplexEq8digits, a, ""){
//	*result_listener << "where the remainder is " << (arg % n);
return areEqualToNDigits(a,arg,8);
}

MATCHER_P (FloatEq8digits, a, ""){
return areEqualToNDigits(a, arg, 8);
}

MATCHER_P2 (FloatEqNdigits, a, n, ""){
return areEqualToNDigits(a, arg, n);
}

		MATCHER_P4 (EqMatrixComplexElementAt, array, i,j, n,std::string("element at (") + std::to_string(i) +
				", " + std::to_string(j) + ") " + (negation ? "is not" : "is") + "(" +
				std::to_string(array[i][j].real) + ", " + std::to_string(array[i][j].imag) + " i ) within " +
				std::to_string(n) + " digits"){
return areEqualToNDigits(arg, array[i][j], n);
}

		MATCHER_P3 (EqArrayComplexElementAt, array, i,n,std::string("element at (") + std::to_string(i) + ") " + (negation ? "is not" : "is") + "(" +
				std::to_string(array[i].real) + ", " + std::to_string(array[i].imag) + " i ) within " +
				std::to_string(n) + " digits"){
return areEqualToNDigits(arg, array[i], n);
}

		MATCHER_P4 (EqMatrixElementAt, array, i,j, n,std::string("element at (") + std::to_string(i) +
				", " + std::to_string(j) + ") " + (negation ? "is not" : "is") +
				std::to_string(array[i][j]) +" within " +
				std::to_string(n) + " digits"){
return areEqualToNDigits(arg, array[i][j], n);
}

::std::ostream& operator<<(::std::ostream& os, const COMPLEX_TYPE& c) {
	return os << c.real << (c.imag < 0 ? "" : " + ") << c.imag << "i";
}

class NoJumpRndNumProvider: public RndNumProvider {
public:
	void initBuffer(int streamId, FLOAT_TYPE *buffer, int bufferSize) override {
		buffer[0] = 0.0;
	}
} NO_JUMP_RND_NUM_PROVIDER;

#endif /* SRC_DEFINITIONS_H_ */
