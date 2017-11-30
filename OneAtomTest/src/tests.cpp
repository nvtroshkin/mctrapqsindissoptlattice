#include <iostream>
#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "mkl.h"

#include <ModelBuilder.h>
#include <precision-definition.h>
#include <Solver.h>
#include <cmath>
#include <ImpreciseValue.h>
#include <MonteCarloSimulator.h>
#include <RndNumProviderImpl.h>
#include <utilities.h>

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

MATCHER_P (ComplexEq8digits, a, ""){
//	*result_listener << "where the remainder is " << (arg % n);
return (a.real == arg.real || std::abs(a.real - arg.real)/std::max(a.real, arg.real)< 10e-9)
&& (a.imag == arg.imag || std::abs(a.imag - arg.imag)/std::max(a.imag, arg.imag)< 10e-9);
}

MATCHER_P (FloatEq8digits, a, ""){
return (a == arg || std::abs(a - arg)/std::max(a, arg)< 10e-9);
}

MATCHER_P2 (FloatEqNdigits, a, n, ""){
return (a == arg || std::abs(a - arg)/std::max(a, arg)< pow(10,-n));
}

::std::ostream& operator<<(::std::ostream& os, const COMPLEX_TYPE& c) {
	return os << c.real << (c.imag < 0 ? " - " : " + ") << c.imag << "i";
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
 *	maxIndex = 3 (the basis size = 4)
 *
 *	timeStep = 0.1
 *	iMax = 1 (total steps)
 *	Rand_seed = 345777
 *
 *	KAPPA = 1.0f;
 *	DELTA_OMEGA = 20.0f;
 *	G = 50.0f;
 *	LATIN_E = 2.0f;
 *
 */
TEST (solver, one_step) {
	std::ostringstream output;

	ModelBuilder modelBuilder(1, 4, 1.0f, 20.0f, 50.0f, 2.0f);
	RndNumProviderImpl rndNumProvider(345777, 1);

	//the ground state
	COMPLEX_TYPE initialState[4] = { { 1.0f, 0.0f } };
	//the previous step vector
	COMPLEX_TYPE resultState[4];

	Solver solver((MKL_INT) 4, 0.1, 1, modelBuilder, rndNumProvider);
	solver.solve(output, initialState, resultState);

	Matcher<COMPLEX_TYPE> matchers[] = { ComplexEq8digits(
	COMPLEX_TYPE { 0.4901174757, -0.006032898977 }), ComplexEq8digits(
	COMPLEX_TYPE { 0.5054537399, -0.3056139613 }), ComplexEq8digits(
	COMPLEX_TYPE { -0.5385949942, 0.3457525846 }), ComplexEq8digits(
	COMPLEX_TYPE { -0.03175209988, 0.01508224744 }) };

	std::vector<COMPLEX_TYPE> cs(4);
	cs.assign(resultState, resultState + 4);
	ASSERT_THAT(cs, ElementsAreArray(matchers))/*<< output.str()*/;
}

/**
 *	nMax = 1
 *	maxIndex = 3 (the basis size = 4)
 *
 *	timeStep = 0.1
 *	iMax = 10 (total steps)
 *	Rand_seed = 345777
 *
 *	KAPPA = 1.0f;
 *	DELTA_OMEGA = 20.0f;
 *	G = 50.0f;
 *	LATIN_E = 2.0f;
 *
 */
TEST (solver, ten_steps_to_check_norm) {
	std::ostringstream output;

	ModelBuilder modelBuilder(1, 4, 1.0f, 20.0f, 50.0f, 2.0f);
	RndNumProviderImpl rndNumProvider(345777, 1);

	//the ground state
	COMPLEX_TYPE initialState[4] = { { 1.0f, 0.0f } };
	//the previous step vector
	COMPLEX_TYPE resultState[4];

	Solver solver((MKL_INT) 4, 0.1, 10, modelBuilder, rndNumProvider);
	solver.solve(output, initialState, resultState);

	Matcher<COMPLEX_TYPE> matchers[] = { ComplexEq8digits(
	COMPLEX_TYPE { 0.01425062492, 0.0142343055 }), ComplexEq8digits(
	COMPLEX_TYPE { 0.5012350831, 0.4978320441 }), ComplexEq8digits(
	COMPLEX_TYPE { -0.4958687013, -0.5024254191 }), ComplexEq8digits(
	COMPLEX_TYPE { -0.03275475156, -0.03363044936 }) };

	std::vector<COMPLEX_TYPE> cs(4);
	cs.assign(resultState, resultState + 4);
	ASSERT_THAT(cs, ElementsAreArray(matchers))/*<< output.str()*/;
}

/**
 *	nMax = 1
 *	maxIndex = 3 (the basis size = 4)
 *
 *	timeStep = 0.001
 *	iMax = 1000 (total steps)
 *	Rand_seed = 345777
 *
 *	KAPPA = 1.0f;
 *	DELTA_OMEGA = 20.0f;
 *	G = 50.0f;
 *	LATIN_E = 2.0f;
 *
 */
TEST (solver, thousand_steps) {
	std::ostringstream output;

	ModelBuilder modelBuilder(1, 4, 1.0f, 20.0f, 50.0f, 2.0f);
	RndNumProviderImpl rndNumProvider(345777, 1);

	//the ground state
	COMPLEX_TYPE initialState[4] = { { 1.0f, 0.0f } };
	//the previous step vector
	COMPLEX_TYPE resultState[4];

	Solver solver((MKL_INT) 4, 0.001, 1000, modelBuilder, rndNumProvider);
	solver.solve(output, initialState, resultState);

	Matcher<COMPLEX_TYPE> matchers[] = { ComplexEq8digits(
	COMPLEX_TYPE { 0.9979644298, 0.03915245248 }), ComplexEq8digits(
	COMPLEX_TYPE { -0.03792430273, 0.02450283386 }), ComplexEq8digits(
	COMPLEX_TYPE { -0.01880338262, 0.01147822274 }), ComplexEq8digits(
	COMPLEX_TYPE { -0.003101962822, 0.0007083183807 }) };

	std::vector<COMPLEX_TYPE> cs(4);
	cs.assign(resultState, resultState + 4);
	ASSERT_THAT(cs, ElementsAreArray(matchers))/*<< output.str()*/;
}

/**
 *	nMax = 3
 *	maxIndex = 7 (the basis size = 8)
 *
 *	timeStep = 0.001
 *	iMax = 1000 (total steps)
 *	Rand_seed = 345777
 *
 *	KAPPA = 1.0f;
 *	DELTA_OMEGA = 20.0f;
 *	G = 50.0f;
 *	LATIN_E = 2.0f;
 *
 */
TEST (solver, thousand_steps_nontrivail_basis) {
	std::ostringstream output;

	ModelBuilder modelBuilder(3, 8, 1.0f, 20.0f, 50.0f, 2.0f);
	RndNumProviderImpl rndNumProvider(345777, 1);

	//the ground state
	COMPLEX_TYPE initialState[8] = { { 1.0f, 0.0f } };
	//the previous step vector
	COMPLEX_TYPE resultState[8];

	Solver solver((MKL_INT) 8, 0.001, 1000, modelBuilder, rndNumProvider);
	solver.solve(output, initialState, resultState);

	Matcher<COMPLEX_TYPE> matchers[] = { ComplexEq8digits(
	COMPLEX_TYPE { 0.9975142689, 0.03872295453 }), ComplexEq8digits(
	COMPLEX_TYPE { -0.04764714178, 0.008266152728 }), ComplexEq8digits(
	COMPLEX_TYPE { -0.03145140378, -0.002311774974 }), ComplexEq8digits(
	COMPLEX_TYPE { 0.00470079133, -0.00369308688 }), ComplexEq8digits(
	COMPLEX_TYPE { 0.004061784976, -0.00454321962 }), ComplexEq8digits(
	COMPLEX_TYPE { 0.002552507868, -0.004815132629 }), ComplexEq8digits(
	COMPLEX_TYPE { 0.00250058722, -0.004882949271 }), ComplexEq8digits(
	COMPLEX_TYPE { 0.00008202311951, -0.0001504026246 }) };

	std::vector<COMPLEX_TYPE> cs(8);
	cs.assign(resultState, resultState + 8);
	ASSERT_THAT(cs, ElementsAreArray(matchers))/*<< output.str()*/;
}

/**
 *	nMax = 3
 *	maxIndex = 7 (the basis size = 8)
 *
 *	KAPPA = 1.0f;
 *	DELTA_OMEGA = 20.0f;
 *	G = 50.0f;
 *	LATIN_E = 2.0f;
 *
 *	Random numbers are such to cause several jumps
 *	etta = 0.99169, 0.964054, 0.667811
 */
TEST (solver, make_jump) {
	std::ostringstream output;

	MKL_INT nMax = 3, basisSize = 8, maxSteps = 10000;
	FLOAT_TYPE timeStep = 0.001, kappa = 1.0, deltaOmega = 20.0, g = 50.0,
			latinE = 2.0;

	ModelBuilder modelBuilder(nMax, basisSize, kappa, deltaOmega, g, latinE);

	class MockRndNumProvider: public RndNumProvider {
	public:
		void initBuffer(int streamId, FLOAT_TYPE *buffer, int bufferSize)
				override {
			buffer[0] = 0.99169;
			buffer[1] = 0.964054;
			buffer[2] = 0.667811;
		}
	} rndNumProviderMock;

	//the ground state
	COMPLEX_TYPE initialState[basisSize] = { { 1.0f, 1.0f }, { 2.0f, 2.0f }, {
			3.0f, 3.0f }, { 4.0f, 4.0f }, { 5.0f, 5.0f }, { 6.0f, 6.0f }, {
			7.0f, 7.0f }, { 8.0f, 8.0f } };
	//the previous step vector
	COMPLEX_TYPE resultState[basisSize];
	//storage for the next random number
	FLOAT_TYPE nextRandom;

	Solver solver(basisSize, timeStep, maxSteps, modelBuilder,
			rndNumProviderMock);
	solver.makeJump(output, nextRandom, initialState, resultState);

//	Matcher<COMPLEX_TYPE> matchers[] = { ComplexEq8digits(
//	COMPLEX_TYPE { 3, 3 }), ComplexEq8digits(
//	COMPLEX_TYPE { 4, 4 }), ComplexEq8digits(
//	COMPLEX_TYPE { 7.071067812, 7.071067812 }), ComplexEq8digits(
//	COMPLEX_TYPE { 8.485281374, 8.485281374 }), ComplexEq8digits(
//	COMPLEX_TYPE { 12.12435565, 12.12435565 }), ComplexEq8digits(
//	COMPLEX_TYPE { 13.85640646, 13.85640646 }), ComplexEq8digits(
//	COMPLEX_TYPE { 0.0, 0.0 }), ComplexEq8digits(
//	COMPLEX_TYPE { 0.0, 0.0 }) };

//normalized
	Matcher<COMPLEX_TYPE> matchers[] = { ComplexEq8digits(
	COMPLEX_TYPE { 0.09622504486493763, 0.09622504486493763 }),
			ComplexEq8digits(
			COMPLEX_TYPE { 0.1283000598, 0.1283000598 }), ComplexEq8digits(
			COMPLEX_TYPE { 0.2268046058, 0.2268046058 }), ComplexEq8digits(
			COMPLEX_TYPE { 0.272165527, 0.272165527 }), ComplexEq8digits(
			COMPLEX_TYPE { 0.3888888889, 0.3888888889 }), ComplexEq8digits(
			COMPLEX_TYPE { 0.4444444444, 0.4444444444 }), ComplexEq8digits(
			COMPLEX_TYPE { 0.0, 0.0 }), ComplexEq8digits(
			COMPLEX_TYPE { 0.0, 0.0 }) };

	std::vector<COMPLEX_TYPE> cs(basisSize);
	cs.assign(resultState, resultState + basisSize);
	ASSERT_THAT(cs, ElementsAreArray(matchers))/*<< output.str()*/;
	ASSERT_THAT(nextRandom, FloatEq8digits(0.99169)); //the first number of the sequence
}

/**
 *	nMax = 3
 *	maxIndex = 7 (the basis size = 8)
 */
TEST (solver, normalize) {
	std::ostringstream output;

	MKL_INT nMax = 3, basisSize = 8, maxSteps = 1000;
	FLOAT_TYPE timeStep = 0.001, kappa = 1.0, deltaOmega = 20.0, g = 50.0,
			latinE = 2.0;

	ModelBuilder modelBuilder(nMax, basisSize, kappa, deltaOmega, g, latinE);
	RndNumProviderImpl rndNumProvider(345777, 1);

	//the ground state
	COMPLEX_TYPE initialState[basisSize] = { { 1.0f, 1.0f }, { 2.0f, 2.0f }, {
			3.0f, 3.0f }, { 4.0f, 4.0f }, { 5.0f, 5.0f }, { 6.0f, 6.0f }, {
			7.0f, 7.0f }, { 8.0f, 8.0f } };

	Solver solver(basisSize, timeStep, maxSteps, modelBuilder, rndNumProvider);
	solver.normalizeVector(initialState);

//normalized
	Matcher<COMPLEX_TYPE> matchers[] = { ComplexEq8digits(
	COMPLEX_TYPE { 0.04950737715, 0.04950737715 }),
			ComplexEq8digits(
			COMPLEX_TYPE { 0.09901475429766744, 0.09901475429766744 }), ComplexEq8digits(
			COMPLEX_TYPE { 0.14852213144650114, 0.14852213144650114 }), ComplexEq8digits(
			COMPLEX_TYPE { 0.1980295086, 0.1980295086 }), ComplexEq8digits(
			COMPLEX_TYPE { 0.2475368857, 0.2475368857 }), ComplexEq8digits(
			COMPLEX_TYPE { 0.2970442629, 0.2970442629 }), ComplexEq8digits(
			COMPLEX_TYPE { 0.34655164004183603, 0.34655164004183603 }), ComplexEq8digits(
			COMPLEX_TYPE { 0.3960590172, 0.3960590172 }) };

	std::vector<COMPLEX_TYPE> cs(basisSize);
	cs.assign(initialState, initialState + basisSize);
	ASSERT_THAT(cs, ElementsAreArray(matchers))/*<< output.str()*/;
}

/**
 *	nMax = 1
 *	maxIndex = 3 (the basis size = 4)
 *
 *	timeStep = 0.1
 *	iMax = 1 (total steps)
 *	Rand_seed = 345777
 *	samples = 1
 *
 *	KAPPA = 1.0f;
 *	DELTA_OMEGA = 20.0f;
 *	G = 50.0f;
 *	LATIN_E = 2.0f;
 *
 */
TEST (simulator, one_sample) {
	std::ostringstream output;

	ModelBuilder modelBuilder(1, 4, 1.0f, 20.0f, 50.0f, 2.0f);
	RndNumProviderImpl rndNumProvider(345777, 1);
	Solver solver((MKL_INT) 4, 0.1, 1, modelBuilder, rndNumProvider);
	MonteCarloSimulator monteCarloSimulator(4, 1, solver);

	SimulationResult *result = monteCarloSimulator.simulate(output);
	ImpreciseValue photonNumber = result->getMeanPhotonNumber();

	ASSERT_THAT(photonNumber.mean, FloatEq8digits(0.4108650876))/*<< output.str()*/;
	ASSERT_THAT(photonNumber.standardDeviation, FloatEq8digits(0.0));

	delete result;
}

/**
 *	nMax = 1
 *	maxIndex = 3 (the basis size = 4)
 *
 *	timeStep = 0.001
 *	iMax = 1000 (total steps)
 *	Rand_seed = 345777
 *	samples = 1
 *
 *	KAPPA = 1.0f;
 *	DELTA_OMEGA = 20.0f;
 *	G = 50.0f;
 *	LATIN_E = 2.0f;
 *
 */
TEST (simulator, one_sample_thousand_steps) {
	std::ostringstream output;

	ModelBuilder modelBuilder(1, 4, 1.0f, 20.0f, 50.0f, 2.0f);
	RndNumProviderImpl rndNumProvider(345777, 1);
	Solver solver((MKL_INT) 4, 0.001, 1000, modelBuilder, rndNumProvider);
	MonteCarloSimulator monteCarloSimulator(4, 1, solver);

	SimulationResult *result = monteCarloSimulator.simulate(output);
	ImpreciseValue photonNumber = result->getMeanPhotonNumber();

	//NDSolve uses less accuracy digits, increasing them solves the difference problem
	ASSERT_THAT(photonNumber.mean, FloatEqNdigits(0.0004954357997,4))/*<< output.str()*/;
	ASSERT_THAT(photonNumber.standardDeviation, FloatEq8digits(0.0));

	delete result;
}

/**
 *	nMax = 3
 *	maxIndex = 7 (the basis size = 8)
 *
 *	timeStep = 0.001
 *	iMax = 1000 (total steps)
 *	Rand_seed = 345777
 *	samples = 1
 *
 *	KAPPA = 1.0f;
 *	DELTA_OMEGA = 20.0f;
 *	G = 50.0f;
 *	LATIN_E = 2.0f;
 *
 */
TEST (simulator, one_sample_thousand_steps_larger_basis) {
	std::ostringstream output;

	MKL_INT nMax = 3, basisSize = 8, maxSteps = 1000, randSeed = 345777,
			nSamples = 1;
	FLOAT_TYPE timeStep = 0.001, kappa = 1.0, deltaOmega = 20.0, g = 50.0,
			latinE = 2.0;

	ModelBuilder modelBuilder(nMax, basisSize, kappa, deltaOmega, g, latinE);
	RndNumProviderImpl rndNumProvider(randSeed, 1);
	Solver solver(basisSize, timeStep, maxSteps, modelBuilder, rndNumProvider);
	MonteCarloSimulator monteCarloSimulator(basisSize, nSamples, solver);

	SimulationResult *result = monteCarloSimulator.simulate(output);
	ImpreciseValue photonNumber = result->getMeanPhotonNumber();

	ASSERT_THAT(photonNumber.mean, FloatEq8digits(0.00125432735))/*<< output.str()*/;
	ASSERT_THAT(photonNumber.standardDeviation, FloatEq8digits(0.0));

	delete result;
}

/**
 *	nMax = 1
 *	maxIndex = 3 (the basis size = 4)
 *
 *	timeStep = 0.001
 *	iMax = 1000 (total steps)
 *	Rand_seed = 345777
 *	samples = 100
 *
 *	KAPPA = 1.0f;
 *	DELTA_OMEGA = 20.0f;
 *	G = 50.0f;
 *	LATIN_E = 2.0f;
 *
 */
TEST (simulator, hundred_samples) {
	std::ostringstream output;

	ModelBuilder modelBuilder(1, 4, 1.0f, 20.0f, 50.0f, 2.0f);
	RndNumProviderImpl rndNumProvider(345777, 1);
	Solver solver((MKL_INT) 4, 0.001, 1000, modelBuilder, rndNumProvider);
	MonteCarloSimulator monteCarloSimulator(4, 100, solver);

	SimulationResult *result = monteCarloSimulator.simulate(output);
	ImpreciseValue photonNumber = result->getMeanPhotonNumber();

	//comparing with NDSolve - its accuracy goal is not enough for timeStep 1/10000
	ASSERT_THAT(photonNumber.mean, FloatEqNdigits(0.0004954357997,4))/*<< output.str()*/;
	ASSERT_THAT(photonNumber.standardDeviation, FloatEqNdigits(8.27328e-13, 6));

	delete result;
}

/**
 *	nMax = 3
 *	maxIndex = 7 (the basis size = 8)
 *
 *	timeStep = 0.001
 *	iMax = 10000 (total steps)
 *	Rand_seed = 345777
 *	samples = 100
 *
 *	KAPPA = 1.0f;
 *	DELTA_OMEGA = 20.0f;
 *	G = 50.0f;
 *	LATIN_E = 2.0f;
 *
 */
TEST (simulator, total) {
	std::ostringstream output;

	MKL_INT nMax = 3, basisSize = 8, maxSteps = 10000, randSeed = 345777,
			nSamples = 100;
	FLOAT_TYPE timeStep = 0.001, kappa = 1.0, deltaOmega = 20.0, g = 50.0,
			latinE = 2.0;

	ModelBuilder modelBuilder(nMax, basisSize, kappa, deltaOmega, g, latinE);
	RndNumProviderImpl rndNumProvider(randSeed, 1);
	Solver solver(basisSize, timeStep, maxSteps, modelBuilder, rndNumProvider);
	MonteCarloSimulator monteCarloSimulator(basisSize, nSamples, solver);

	SimulationResult *result = monteCarloSimulator.simulate(output);
	ImpreciseValue photonNumber = result->getMeanPhotonNumber();

	output.precision(10);
	output << "mean photon number: " << photonNumber.mean << std::endl;
	output << "standard deviation: " << photonNumber.standardDeviation << std::endl;

	//in a jump occurs a significant loss of precision, because
	// 1) the method of detection of the time of a jump is rough (just using a previous step)
	// 2) after a jump a state vector gets 1/100 of its previous norm with 2 significant digits cut
	ASSERT_THAT(photonNumber.mean, FloatEqNdigits(0.001441730225, 8))/*<< output.str()*/;
	ASSERT_THAT(photonNumber.standardDeviation, FloatEqNdigits(0.001053123598, 8));

	delete result;
}

int main(int argc, char **argv) {
	printf("Running main() from gtest_main.cc\n");
	printf("Work only for n=1\n");
	testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
