/*
 * misc.h
 *
 *  Created on: Dec 8, 2017
 *      Author: fakesci
 */

#ifndef SRC_DEFINITIONS_H_
#define SRC_DEFINITIONS_H_

#include "helper_cuda.h"

#include <cmath>
#include <iostream>
#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "precision-definition.h"
#include <string>

using namespace testing;

#if defined(SINGLE_PRECISION)
static const int RIGHT_DIGITS = 6;
#elif defined(DOUBLE_PRECISION)
static const int RIGHT_DIGITS = 14;
#endif

#define ASSERT_COMPLEX(c, expReal, expImagine)\
{											\
	ASSERT_FLOAT_EQ(c.x, expReal);\
	ASSERT_FLOAT_EQ(c.y, expImagine);\
}

#define FLOAT_PRECISION 12

struct FancyStream: public std::ostringstream {
	FancyStream() {
		(*this) << std::scientific << std::setprecision(FLOAT_PRECISION);
	}
};

MATCHER_P (ComplexNumberEquals, a, "") {
//	*result_listener << "where the remainder is " << (arg % n);
	return a.x == arg.x && a.y == arg.y;
}

inline int getDecimalExponent(FLOAT_TYPE f) {
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

inline bool areEqualToNDigits(FLOAT_TYPE a, FLOAT_TYPE b, int nDigits) {
	if (a == b) {
		return true;
	}

	FLOAT_TYPE diff = std::abs(a - b);
	FLOAT_TYPE max = std::max(std::abs(a), std::abs(b));

	int diffExp = getDecimalExponent(diff), maxExp = getDecimalExponent(max);

	return diffExp <= maxExp && std::abs(maxExp - diffExp) >= nDigits;
}

inline bool areEqualToNDigits(CUDA_COMPLEX_TYPE a, CUDA_COMPLEX_TYPE b,
		int nDigits) {
	//compare modules
//	return areEqualToNDigits(a.x * a.x + a.y*a.y, b.x * b.x + b.y*b.y, nDigits);
	return areEqualToNDigits(a.x, b.x, nDigits)
			&& areEqualToNDigits(a.y, b.y, nDigits);
}

MATCHER_P (ComplexEq8digits, a, "") {
//	*result_listener << "where the remainder is " << (arg % n);
	return areEqualToNDigits(a,arg,8);
}

MATCHER_P (FloatEq8digits, a, "") {
	return areEqualToNDigits(a, arg, 8);
}

MATCHER_P2 (FloatEqNdigits, a, n, "") {
	return areEqualToNDigits(a, arg, n);
}

MATCHER_P4 (EqMatrixComplexElementAt, array, i,j, n,
		dynamic_cast<std::ostringstream&>(FancyStream() << std::string("element at (") << i <<
				", " << j << ") " << (negation ? "is not" : "is") << "(" <<
				array[i][j].x << ", " << array[i][j].y << " i ) within " <<
				n << " digits").str()) {
	return areEqualToNDigits(arg, array[i][j], n);
}

MATCHER_P3 (EqArrayComplexElementAt, array, i,n,
		dynamic_cast<std::ostringstream&>(FancyStream() << "element at (" << i << ") " << (negation ? "is not" : "is") << "("
				<< array[i].x << ", " << array[i].y << " i ) within "
				<< n << " digits").str()) {
	return areEqualToNDigits(arg, array[i], n);
}

MATCHER_P4 (EqMatrixElementAt, array, i,j, n,
		dynamic_cast<std::ostringstream&>(FancyStream() << "element at (" <<i
				<<", " <<j << ") " << (negation ? "is not" : "is")
				<<array[i][j]<<" within "
				<<n << " digits").str()) {
	return areEqualToNDigits(arg, array[i][j], n);
}

inline ::std::ostream& operator<<(::std::ostream& os,
		const CUDA_COMPLEX_TYPE& c) {
	return os
			<< dynamic_cast<std::ostringstream&>(FancyStream() << c.x
					<< (c.y < 0 ? "" : " + ") << c.y << "i").str();
}

void _checkState(const std::string &caseId, uint basisSize,
		const CUDA_COMPLEX_TYPE * const resultState,
		const CUDA_COMPLEX_TYPE * const expectedState, uint rightDigits) {
	//check the matrix
	for (int i = 0; i < basisSize; ++i) {
		ASSERT_THAT(resultState[i],
				EqArrayComplexElementAt(expectedState, i, rightDigits))
				<< caseId << std::endl;
	}
}

void _checkDeviceState(const std::string &caseId, const uint basisSize,
		const CUDA_COMPLEX_TYPE * const actualResultDevPtr,
		const CUDA_COMPLEX_TYPE * const expectedState, const uint rightDigits) {

	CUDA_COMPLEX_TYPE * actualResult = new CUDA_COMPLEX_TYPE[basisSize];

	checkCudaErrors(
			cudaMemcpy(actualResult, actualResultDevPtr,
					basisSize * sizeof(CUDA_COMPLEX_TYPE),
					cudaMemcpyDeviceToHost));

	_checkState(caseId, basisSize, actualResult, expectedState, rightDigits);

	delete[] actualResult;
}

//class NoJumpRndNumProvider: public RndNumProvider {
//public:
//	void initBuffer(int streamId, FLOAT_TYPE *buffer, int bufferSize) override {
//		buffer[0] = 0.0;
//	}
//} NO_JUMP_RND_NUM_PROVIDER;

#endif /* SRC_DEFINITIONS_H_ */
