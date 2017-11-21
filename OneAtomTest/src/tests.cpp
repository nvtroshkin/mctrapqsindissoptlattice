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

MATCHER_P (ComplexNumberEquals, a, ""){
//	*result_listener << "where the remainder is " << (arg % n);
return a.real == arg.real && a.imag == arg.imag;
}

::std::ostream& operator<<(::std::ostream& os, const MKL_Complex8& c) {
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
	MatrixDiagForm diagH = getHhatInDiagForm();

	ASSERT_EQ(diagH.diagDistLength, 5);

	std::vector<int> *vec = new std::vector<int>();
	(*vec).assign(diagH.diagsDistances,
			diagH.diagsDistances + diagH.diagDistLength);
	ASSERT_THAT(*vec, ElementsAre(-2, -1, 0, 1, 2));

	std::vector<MKL_Complex8> *cs = new std::vector<MKL_Complex8>();
	(*cs).assign(diagH.matrix,
			diagH.matrix + diagH.diagDistLength * diagH.leadDimension);

	Matcher<MKL_Complex8> matchers[] = { A<MKL_Complex8>(), A<MKL_Complex8>(),
			ComplexNumberEquals(MKL_Complex8 { 0, 0 }), ComplexNumberEquals(
					MKL_Complex8 { 0, 0 }), ComplexNumberEquals(MKL_Complex8 {
					0, -2 }), A<MKL_Complex8>(), ComplexNumberEquals(
					MKL_Complex8 { 0, 0 }), ComplexNumberEquals(MKL_Complex8 {
					0, 20 }), ComplexNumberEquals(MKL_Complex8 { 0, -50 }),
			ComplexNumberEquals(MKL_Complex8 { 0, -2 }), ComplexNumberEquals(
					MKL_Complex8 { 0, -2 }), ComplexNumberEquals(MKL_Complex8 {
					0, -50 }), ComplexNumberEquals(MKL_Complex8 { -1, 20 }),
			ComplexNumberEquals(MKL_Complex8 { 0, 0 }), A<MKL_Complex8>(),
			ComplexNumberEquals(MKL_Complex8 { 0, -2 }), ComplexNumberEquals(
					MKL_Complex8 { 0, 0 }), ComplexNumberEquals(MKL_Complex8 {
					-1, 40 }), A<MKL_Complex8>(), A<MKL_Complex8>() };
	ASSERT_THAT(*cs, ElementsAreArray(matchers));
}

/**
 * H=
 *
 * (0	0	-2i		0)
 * (0	20i	-50i	-2i)
 * (-2i	-50i	-1+20i	0)
 * (0	-2i		0	-1+40i)
 *
 * values: {0, 0, -2i, 0, 20i, -50i, -2i, -2i, -50i, -1+20i, 0, -2i, 0, -1+40i}
 * columns: {0,1,2,0,1,2,3,0,1,2,3,1,2,3}
 * rowIndex: {0,3,7,11,14}
 *
 */
TEST (H_CSR3_form_test, ALL) {
	CSR3Matrix hCSR3 = getHInCSR3();

	ASSERT_EQ(hCSR3.rowsNumber, 4);

	//check the total values number
	ASSERT_EQ(hCSR3.rowIndex[4], 14);

	std::vector<int> *vec = new std::vector<int>();
	(*vec).assign(hCSR3.columns, hCSR3.columns + 14);
	ASSERT_THAT(*vec, ElementsAreArray( { 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 1, 2,
			3 }));

	(*vec).assign(hCSR3.rowIndex, hCSR3.rowIndex + 5);
	ASSERT_THAT(*vec, ElementsAreArray( {0,3,7,11,14}));

	std::vector<MKL_Complex8> *cs = new std::vector<MKL_Complex8>();
	(*cs).assign(hCSR3.values, hCSR3.values + 14);

	Matcher<MKL_Complex8> matchers[] = { ComplexNumberEquals(
			MKL_Complex8 { 0, 0 }), ComplexNumberEquals(MKL_Complex8 { 0, 0 }),
			ComplexNumberEquals(MKL_Complex8 { 0, -2 }), ComplexNumberEquals(
					MKL_Complex8 { 0, 0 }), ComplexNumberEquals(MKL_Complex8 {
					0, 20 }), ComplexNumberEquals(MKL_Complex8 { 0, -50 }), ComplexNumberEquals(
					MKL_Complex8 { 0, -2 }), ComplexNumberEquals(MKL_Complex8 {
					0, -2 }), ComplexNumberEquals(MKL_Complex8 { 0, -50 }),
			ComplexNumberEquals(MKL_Complex8 { -1, 20 }), ComplexNumberEquals(
					MKL_Complex8 { 0, 0 }), ComplexNumberEquals(MKL_Complex8 {
					0, -2 }), ComplexNumberEquals(MKL_Complex8 { 0, 0 }),
			ComplexNumberEquals(MKL_Complex8 { -1, 40 })};
	ASSERT_THAT(*cs, ElementsAreArray(matchers));
}

/**
 * aPlus =
 *
 * (0	0	0	0)
 * (0	0	0	0)
 * (1	0	0	0)
 * (0	1	0	0)
 *
 * values: {0,0,1,1}
 * columns: {0,0,0,1}
 * rowIndex: {0,1,2,3,4}
 *
 */
TEST (aPlus_CSR3_form_test, ALL) {
	CSR3Matrix aPlusCSR3 = getAPlusInCSR3();

	ASSERT_EQ(aPlusCSR3.rowsNumber, 4);

	//check the total values number
	ASSERT_EQ(aPlusCSR3.rowIndex[4], 4);

	std::vector<int> *vec = new std::vector<int>();
	(*vec).assign(aPlusCSR3.columns, aPlusCSR3.columns + 4);
	ASSERT_THAT(*vec, ElementsAre(0,0,0,1));

	(*vec).assign(aPlusCSR3.rowIndex, aPlusCSR3.rowIndex + 5);
	ASSERT_THAT(*vec, ElementsAre(0,1,2,3,4));

	std::vector<MKL_Complex8> *cs = new std::vector<MKL_Complex8>();
	(*cs).assign(aPlusCSR3.values, aPlusCSR3.values + 4);

	Matcher<MKL_Complex8> matchers[] = { ComplexNumberEquals(
			MKL_Complex8 { 0, 0 }), ComplexNumberEquals(MKL_Complex8 { 0, 0 }),
			ComplexNumberEquals(MKL_Complex8 { 1, 0 }), ComplexNumberEquals(
					MKL_Complex8 { 1, 0 })};
	ASSERT_THAT(*cs, ElementsAreArray(matchers));
}

/**
 * a =
 *
 * (0	0	1	0)
 * (0	0	0	1)
 * (0	0	0	0)
 * (0	0	0	0)
 *
 * values: {1,1,0,0}
 * columns: {2,3,3,3}
 * rowIndex: {0,1,2,3,4}
 *
 */
TEST (a_CSR3_form_test, ALL) {
	CSR3Matrix aCSR3 = getAInCSR3();

	ASSERT_EQ(aCSR3.rowsNumber, 4);

	//check the total values number
	ASSERT_EQ(aCSR3.rowIndex[4], 4);

	std::vector<int> *vec = new std::vector<int>();
	(*vec).assign(aCSR3.columns, aCSR3.columns + 4);
	ASSERT_THAT(*vec, ElementsAre(2,3,3,3));

	(*vec).assign(aCSR3.rowIndex, aCSR3.rowIndex + 5);
	ASSERT_THAT(*vec, ElementsAre(0,1,2,3,4));

	std::vector<MKL_Complex8> *cs = new std::vector<MKL_Complex8>();
	(*cs).assign(aCSR3.values, aCSR3.values + 4);

	Matcher<MKL_Complex8> matchers[] = { ComplexNumberEquals(
			MKL_Complex8 { 1, 0 }), ComplexNumberEquals(MKL_Complex8 { 01, 0 }),
			ComplexNumberEquals(MKL_Complex8 { 0, 0 }), ComplexNumberEquals(
					MKL_Complex8 { 0, 0 })};
	ASSERT_THAT(*cs, ElementsAreArray(matchers));
}

int main(int argc, char **argv) {
	printf("Running main() from gtest_main.cc\n");
	printf("Work only for n=1\n");
	testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
