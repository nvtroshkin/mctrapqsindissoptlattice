#include <iostream>
#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "mkl.h"

#include <macro.h>
#include <ModelBuilder.h>
#include <Solver.h>

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

::std::ostream& operator<<(::std::ostream& os, const COMPLEX_TYPE& c) {
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
	const FLOAT_TYPE EXPECTED_RESULT[4][4][2] = { { { 0.0f, 0.0f },
			{ 0.0f, 0.0f }, { 0.0f, -2.0f }, { 0.0f, 0.0f } }, { { 0.0f, 0.0f },
			{ 0.0f, 20.0f }, { 0.0f, -50.0f }, { 0.0f, -2.0f } }, { { 0.0f,
			-2.0f }, { 0.0f, -50.0f }, { -1.0f, 20.0f }, { 0.0f, 0.0f } }, { {
			0.0f, 0.0f }, { 0.0f, -2.0f }, { 0.0f, 0.0f }, { -1.0f, 40.0f } } };

	ModelBuilder modelBuilder(1, 4, 1.0f, 20.0f, 50.0f, 2.0f);

	//check the matrix
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			COMPLEX_TYPE c = modelBuilder.H(i, j);
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
	const FLOAT_TYPE EXPECTED_RESULT[4][4] = { { 0.0f, 0.0f, 0.0f, 0.0f }, {
			0.0f, 0.0f, 0.0f, 0.0f }, { 1.0f, 0.0f, 0.0f, 0.0f }, { 0.0f, 1.0f,
			0.0f, 0.0f } };

	ModelBuilder modelBuilder(1, 4, 1.0f, 20.0f, 50.0f, 2.0f);

	//check the matrix
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			FLOAT_TYPE result = modelBuilder.aPlus(i, j);
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
	const FLOAT_TYPE EXPECTED_RESULT[4][4] = { { 0.0f, 0.0f, 0.0f, 0.0f }, {
			1.0f, 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f,
			1.0f, 0.0f } };

	ModelBuilder modelBuilder(1, 4, 1.0f, 20.0f, 50.0f, 2.0f);

	//check the matrix
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			FLOAT_TYPE result = modelBuilder.sigmaPlus(i, j);
			ASSERT_FLOAT_EQ(result, EXPECTED_RESULT[i][j]);
		}
	}
}

///**
// * H=
// *
// * (0	0	-2i		0)
// * (0	20i	-50i	-2i)
// * (-2i	-50i	-1+20i	0)
// * (0	-2i		0	-1+40i)
// *
// * distances: -2 -1 0 1 2
// *
// * diagonals:
// * (x	x		0		0		-2i)
// * (x	0		20i		-50i	-2i)
// * (-2i	-50i	-1+20i	0		x)
// * (-2i 0		-1+40i	x		x)
// *
// * x - an arbitrary number
// *
// */
//TEST (H_diagonal_form_test, ALL) {
//	MatrixDiagForm diagH = getHhatInDiagForm();
//
//	ASSERT_EQ(diagH.diagDistLength, 5);
//
//	std::vector<int> *vec = new std::vector<int>();
//	(*vec).assign(diagH.diagsDistances,
//			diagH.diagsDistances + diagH.diagDistLength);
//	ASSERT_THAT(*vec, ElementsAre(-2, -1, 0, 1, 2));
//
//	std::vector<COMPLEX_TYPE> *cs = new std::vector<COMPLEX_TYPE>();
//	(*cs).assign(diagH.matrix,
//			diagH.matrix + diagH.diagDistLength * diagH.leadDimension);
//
//	Matcher<COMPLEX_TYPE> matchers[] = { A<COMPLEX_TYPE>(), A<COMPLEX_TYPE>(),
//			ComplexNumberEquals(COMPLEX_TYPE { 0, 0 }), ComplexNumberEquals(
//			COMPLEX_TYPE { 0, 0 }), ComplexNumberEquals(COMPLEX_TYPE { 0, -2 }),
//			A<COMPLEX_TYPE>(), ComplexNumberEquals(
//			COMPLEX_TYPE { 0, 0 }), ComplexNumberEquals(COMPLEX_TYPE { 0, 20 }),
//			ComplexNumberEquals(COMPLEX_TYPE { 0, -50 }), ComplexNumberEquals(
//					COMPLEX_TYPE { 0, -2 }), ComplexNumberEquals(
//			COMPLEX_TYPE { 0, -2 }), ComplexNumberEquals(
//					COMPLEX_TYPE { 0, -50 }), ComplexNumberEquals(COMPLEX_TYPE {
//					-1, 20 }), ComplexNumberEquals(COMPLEX_TYPE { 0, 0 }), A<
//					COMPLEX_TYPE>(), ComplexNumberEquals(
//					COMPLEX_TYPE { 0, -2 }), ComplexNumberEquals(
//			COMPLEX_TYPE { 0, 0 }), ComplexNumberEquals(
//					COMPLEX_TYPE { -1, 40 }), A<COMPLEX_TYPE>(),
//			A<COMPLEX_TYPE>() };
//	ASSERT_THAT(*cs, ElementsAreArray(matchers));
//}

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
	ModelBuilder modelBuilder(1, 4, 1.0f, 20.0f, 50.0f, 2.0f);

	CSR3Matrix *hCSR3 = modelBuilder.getHInCSR3();

	ASSERT_EQ(hCSR3->rowsNumber, 4);

	//check the total values number
	ASSERT_EQ(hCSR3->rowIndex[4], 14);

	std::vector<int> vec(14);
	vec.assign(hCSR3->columns, hCSR3->columns + 14);
	ASSERT_THAT(vec, ElementsAreArray( { 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 1, 2,
			3 }));

	vec.assign(hCSR3->rowIndex, hCSR3->rowIndex + 5);
	ASSERT_THAT(vec, ElementsAreArray( { 0, 3, 7, 11, 14 }));

	std::vector<COMPLEX_TYPE> cs(4);
	cs.assign(hCSR3->values, hCSR3->values + 14);

	Matcher<COMPLEX_TYPE> matchers[] = { ComplexNumberEquals(
	COMPLEX_TYPE { 0, 0 }), ComplexNumberEquals(COMPLEX_TYPE { 0, 0 }),
			ComplexNumberEquals(COMPLEX_TYPE { 0, -2 }), ComplexNumberEquals(
			COMPLEX_TYPE { 0, 0 }), ComplexNumberEquals(COMPLEX_TYPE { 0, 20 }),
			ComplexNumberEquals(COMPLEX_TYPE { 0, -50 }), ComplexNumberEquals(
			COMPLEX_TYPE { 0, -2 }), ComplexNumberEquals(
			COMPLEX_TYPE { 0, -2 }), ComplexNumberEquals(
			COMPLEX_TYPE { 0, -50 }), ComplexNumberEquals(
			COMPLEX_TYPE { -1, 20 }), ComplexNumberEquals(
			COMPLEX_TYPE { 0, 0 }), ComplexNumberEquals(COMPLEX_TYPE { 0, -2 }),
			ComplexNumberEquals(COMPLEX_TYPE { 0, 0 }), ComplexNumberEquals(
			COMPLEX_TYPE { -1, 40 }) };
	ASSERT_THAT(cs, ElementsAreArray(matchers));
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
	ModelBuilder modelBuilder(1, 4, 1.0f, 20.0f, 50.0f, 2.0f);

	CSR3Matrix *aPlusCSR3 = modelBuilder.getAPlusInCSR3();

	ASSERT_EQ(aPlusCSR3->rowsNumber, 4);

	//check the total values number
	ASSERT_EQ(aPlusCSR3->rowIndex[4], 4);

	std::vector<int> vec(5);
	vec.assign(aPlusCSR3->columns, aPlusCSR3->columns + 4);
	ASSERT_THAT(vec, ElementsAre(0, 0, 0, 1));

	vec.assign(aPlusCSR3->rowIndex, aPlusCSR3->rowIndex + 5);
	ASSERT_THAT(vec, ElementsAre(0, 1, 2, 3, 4));

	std::vector<COMPLEX_TYPE> cs(4);
	cs.assign(aPlusCSR3->values, aPlusCSR3->values + 4);

	Matcher<COMPLEX_TYPE> matchers[] = { ComplexNumberEquals(
	COMPLEX_TYPE { 0, 0 }), ComplexNumberEquals(COMPLEX_TYPE { 0, 0 }),
			ComplexNumberEquals(COMPLEX_TYPE { 1, 0 }), ComplexNumberEquals(
			COMPLEX_TYPE { 1, 0 }) };
	ASSERT_THAT(cs, ElementsAreArray(matchers));
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
	ModelBuilder modelBuilder(1, 4, 1.0f, 20.0f, 50.0f, 2.0f);

	CSR3Matrix *aCSR3 = modelBuilder.getAInCSR3();

	ASSERT_EQ(aCSR3->rowsNumber, 4);

	//check the total values number
	ASSERT_EQ(aCSR3->rowIndex[4], 4);

	std::vector<int> vec(4);
	vec.assign(aCSR3->columns, aCSR3->columns + 4);
	ASSERT_THAT(vec, ElementsAre(2, 3, 3, 3));

	vec.assign(aCSR3->rowIndex, aCSR3->rowIndex + 5);
	ASSERT_THAT(vec, ElementsAre(0, 1, 2, 3, 4));

	std::vector<COMPLEX_TYPE> cs(4);
	cs.assign(aCSR3->values, aCSR3->values + 4);

	Matcher<COMPLEX_TYPE> matchers[] = { ComplexNumberEquals(
	COMPLEX_TYPE { 1, 0 }), ComplexNumberEquals(COMPLEX_TYPE { 01, 0 }),
			ComplexNumberEquals(COMPLEX_TYPE { 0, 0 }), ComplexNumberEquals(
			COMPLEX_TYPE { 0, 0 }) };
	ASSERT_THAT(cs, ElementsAreArray(matchers));
}

/**
 *	nMax = 1
 *	maxIndex = 3 (the basis size)
 *
 *	timeStep = 0.001
 *	iMax = 1 (total steps)
 *
 *	KAPPA = 1.0f;
 *	DELTA_OMEGA = 20.0f;
 *	G = 50.0f;
 *	LATIN_E = 2.0f;
 *
 */
TEST (solver, one_step) {
	ModelBuilder modelBuilder(1, 4, 1.0f, 20.0f, 50.0f, 2.0f);

	CSR3Matrix *h = modelBuilder.getHInCSR3();
	CSR3Matrix *a = modelBuilder.getAInCSR3();

	//the ground state
	COMPLEX_TYPE step1State[4] = { { 1.0f, 0.0f } };
	//the previous step vector
	COMPLEX_TYPE step2State[4];

	Solver solver((MKL_INT) 3);
	solver.solve(h->values, h->rowIndex, h->columns, a->values, a->rowIndex,
			a->columns, step1State, step2State);

	Matcher<COMPLEX_TYPE> matchers[] = { ComplexNumberEquals(
	COMPLEX_TYPE { 1, 0 }), ComplexNumberEquals(COMPLEX_TYPE { 01, 0 }),
			ComplexNumberEquals(COMPLEX_TYPE { 0, 0 }), ComplexNumberEquals(
			COMPLEX_TYPE { 0, 0 }) };

	std::vector<COMPLEX_TYPE> cs(4);
		cs.assign(step2State, step2State + 4);
	ASSERT_THAT(cs, ElementsAreArray(matchers));
}

int main(int argc, char **argv) {
	printf("Running main() from gtest_main.cc\n");
	printf("Work only for n=1\n");
	testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
