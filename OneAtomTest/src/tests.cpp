#include <iostream>
#include <runge-kutta.h>
#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "mkl.h"

using namespace testing;

#define ASSERT_COMPLEX(c, expReal, expImagine)\
{											\
	ASSERT_FLOAT_EQ(c.real, expReal);\
	ASSERT_FLOAT_EQ(c.imag, expImagine);\
}

MATCHER_P (ComplexNumberEquals, a, "") {
//	*result_listener << "where the remainder is " << (arg % n);
	return a.real == arg.real && a.imag == arg.imag;
}

::std::ostream& operator<<(::std::ostream& os, const MKL_Complex8& c)
{
    return os << c.real << " + " << c.imag << "i";
}

/**
 * H=
 *
 * (0	0	-2i		0)
 * (0	20i	-50i	-2i)
 * (-2i	-50i	-1+20i	0)
 * (0	-2i		0	-1+40i)
 */
TEST (H_tests,ALL) {
	const float EXPECTED_RESULT[4][4][2] = { { { 0.0f, 0.0f }, { 0.0f, 0.0f }, {
			0.0f, -2.0f }, { 0.0f, 0.0f } }, { { 0.0f, 0.0f }, { 0.0f, 20.0f },
			{ 0.0f, -50.0f }, { 0.0f, -2.0f } }, { { 0.0f, -2.0f }, { 0.0f,
			-50.0f }, { -1.0f, 20.0f }, { 0.0f, 0.0f } }, { { 0.0f, 0.0f }, {
			0.0f, -2.0f }, { 0.0f, 0.0f }, { -1.0f, 40.0f } } };
	//init cache
	initPhotonNumbersSqrts();

	//check the matrix
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			MKL_Complex8 c = H(i, j, 4, 1.0f, 20.0f, 50.0f, 2.0f);
			ASSERT_COMPLEX(c, EXPECTED_RESULT[i][j][0],
					EXPECTED_RESULT[i][j][1]);
		}
	}
}

/**
 * aPlus=
 *
 * (0	0	0	0)
 * (0	0	0	0)
 * (1	0	0	0)
 * (0	1	0	0)
 */
TEST (aPlus_test, ALL) {
	const float EXPECTED_RESULT[4][4] = { { 0.0f, 0.0f, 0.0f, 0.0f }, { 0.0f,
			0.0f, 0.0f, 0.0f }, { 1.0f, 0.0f, 0.0f, 0.0f }, { 0.0f, 1.0f, 0.0f,
			0.0f } };
	//init cache...
	initPhotonNumbersSqrts();

	//check the matrix
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			float result = aPlus(i, j);
			ASSERT_FLOAT_EQ(result, EXPECTED_RESULT[i][j]);
		}
	}
}

/**
 * sigmaPlus=
 *
 * (0	0	0	0)
 * (1	0	0	0)
 * (0	0	0	0)
 * (0	0	1	0)
 */
TEST (sigmaPlus_test, ALL) {
	const float EXPECTED_RESULT[4][4] = { { 0.0f, 0.0f, 0.0f, 0.0f }, { 1.0f,
			0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 1.0f,
			0.0f } };
	//init cache...
	initPhotonNumbersSqrts();

	//check the matrix
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			float result = sigmaPlus(i, j);
			ASSERT_FLOAT_EQ(result, EXPECTED_RESULT[i][j]);
		}
	}
}

/**
 * H=
 *
 * (0	0	-2i		0)
 * (0	20i	-50i	-2i)
 * (-2i	-50i	-1+20i	0)
 * (0	-2i		0	-1+40i)
 *
 * distances: -2 -1 0 1 2
 *
 * diagonals:
 * (x	x		0		0		-2i)
 * (x	0		20i		-50i	-2i)
 * (-2i	-50i	-1+20i	0		x)
 * (-2i 0		-1+40i	x		x)
 *
 * x - an arbitrary number
 *
 */
TEST (H_diagonal_form_test, ALL) {
	//a temporary vector
	std::vector<int> *vec = new std::vector<int>();

	MatrixDiagForm diagH = getHhatInDiagForm();

	ASSERT_EQ(diagH.diagDistLength, 5);

	(*vec).assign(diagH.diagsDistances, diagH.diagsDistances + diagH.diagDistLength);
	ASSERT_THAT(*vec, ElementsAre(-2,-1,0,1,2));

//	MKL_Complex8 expectedH[] = {A<MKL_Complex8>(), A<MKL_Complex8>(), };

	MKL_Complex8 c = {1,2};

	ASSERT_THAT(diagH.matrix, ComplexNumberEquals(c));

//	ASSERT_THAT(diagH.matrix, testing::ElementsAreArray({MklComplexEq({1,2})}));
}

int main(int argc, char **argv) {
	printf("Running main() from gtest_main.cc\n");
	testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
