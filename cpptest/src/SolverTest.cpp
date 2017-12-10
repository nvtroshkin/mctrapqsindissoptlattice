/*
 * solver-tests.cpp
 *
 *  Created on: Dec 8, 2017
 *      Author: fakesci
 */

#include "definitions.h"
#include "Model.h"
#include "Solver.h"
#include "RndNumProviderImpl.h"

TEST (Solver, normalize) {
	std::ostringstream output;

	const int basisSize = 8;

	Model model(1, 2, 2, 2, 1.0, 20.0, 50.0, 2.0, 0.1);
	RndNumProviderImpl rndNumProvider(345777, 1);

	//the ground state
	COMPLEX_TYPE initialState[] = { { 1.0f, 1.0f }, { 2.0f, 2.0f },
			{ 3.0f, 3.0f }, { 4.0f, 4.0f }, { 5.0f, 5.0f }, { 6.0f, 6.0f }, {
					7.0f, 7.0f }, { 8.0f, 8.0f } };

	Solver solver(0, 0.1, 1, model, NO_JUMP_RND_NUM_PROVIDER);
	solver.normalizeVector(initialState);

//normalized
	Matcher<COMPLEX_TYPE> matchers[] = { ComplexEq8digits(
	COMPLEX_TYPE { 0.04950737715, 0.04950737715 }), ComplexEq8digits(
	COMPLEX_TYPE { 0.09901475429766744, 0.09901475429766744 }),
			ComplexEq8digits(
			COMPLEX_TYPE { 0.14852213144650114, 0.14852213144650114 }),
			ComplexEq8digits(
			COMPLEX_TYPE { 0.1980295086, 0.1980295086 }), ComplexEq8digits(
			COMPLEX_TYPE { 0.2475368857, 0.2475368857 }), ComplexEq8digits(
			COMPLEX_TYPE { 0.2970442629, 0.2970442629 }), ComplexEq8digits(
			COMPLEX_TYPE { 0.34655164004183603, 0.34655164004183603 }),
			ComplexEq8digits(
			COMPLEX_TYPE { 0.3960590172, 0.3960590172 }) };

	std::vector<COMPLEX_TYPE> cs(basisSize);
	cs.assign(initialState, initialState + basisSize);
	ASSERT_THAT(cs, ElementsAreArray(matchers))/*<< output.str()*/;
}

/**
 *	KAPPA = 1.0
 *	DELTA_OMEGA = 20.0
 *	G = 50.0
 *	LATIN_E = 2.0
 *	J = 0.1
 *
 *	atom1SSize = atom2SSize = 2
 *	field1SSize = field2SSize = 2
 *
 *	timeStep = 0.1
 *	timeStepsNumber = 1
 *
 *	No jumps
 *
 */
TEST (Solver, oneLargeStepNoJumps) {
	std::ostringstream output;

	Model model(2, 2, 2, 2, 1.0, 20.0, 50.0, 2.0, 0.1);

	const int basisSize = 16;

	//the ground state
	COMPLEX_TYPE initialState[basisSize] = { { 1.0, 0.0 } };
	//the previous step vector
	COMPLEX_TYPE resultState[basisSize];

	Solver solver(0, 0.1, 1, model, NO_JUMP_RND_NUM_PROVIDER);
	solver.solve(output, initialState, resultState);

	Matcher<COMPLEX_TYPE> matchers[] = {
			//0
			ComplexEq8digits(COMPLEX_TYPE { 0.002854485509, -5.886219873e-6 }),
			ComplexEq8digits(COMPLEX_TYPE { 0.1959873121, 0.000206464035 }),
			ComplexEq8digits(COMPLEX_TYPE { 0.1875856765, -0.001694597408 }),
			ComplexEq8digits(COMPLEX_TYPE { 0.06143043855, -0.001876989243 }),

			//4
			ComplexEq8digits(COMPLEX_TYPE { 0.1959873121, 0.000206464035 }),
			ComplexEq8digits(COMPLEX_TYPE { 0.3891289667, 0.00041881429 }),
			ComplexEq8digits(COMPLEX_TYPE { 0.380727331, -0.001482247153 }),
			ComplexEq8digits(COMPLEX_TYPE { 0.2545720931, -0.001664638988 }),

			//8
			ComplexEq8digits(COMPLEX_TYPE { 0.1875856765, -0.001694597408 }),
			ComplexEq8digits(COMPLEX_TYPE { 0.380727331, -0.001482247153 }),
			ComplexEq8digits(COMPLEX_TYPE { 0.3723256953, -0.003383308597 }),
			ComplexEq8digits(COMPLEX_TYPE { 0.2461704574, -0.003565700431 }),

			//12
			ComplexEq8digits(COMPLEX_TYPE { 0.06143043855, -0.001876989243 }),
			ComplexEq8digits(COMPLEX_TYPE { 0.2545720931, -0.001664638988 }),
			ComplexEq8digits(COMPLEX_TYPE { 0.2461704574, -0.003565700431 }),
			ComplexEq8digits(COMPLEX_TYPE { 0.1200152195, -0.003748092266 })
	//
			};

	std::vector<COMPLEX_TYPE> cs(basisSize);
	cs.assign(resultState, resultState + basisSize);
	ASSERT_THAT(cs, ElementsAreArray(matchers))/*<< output.str()*/;
}

/**
 *	KAPPA = 1.0
 *	DELTA_OMEGA = 20.0
 *	G = 50.0
 *	LATIN_E = 2.0
 *	J = 0.1
 *
 *	atom1SSize = atom2SSize = 2
 *	field1SSize = field2SSize = 2
 *
 *	timeStep = 0.1
 *	timeStepsNumber = 10
 *
 *	No jumps
 *
 */
TEST (Solver, tenLargeStepsNoJump) {
	std::ostringstream output;

	Model model(2, 2, 2, 2, 1.0, 20.0, 50.0, 2.0, 0.1);

	const int basisSize = 16;

	//the ground state
	COMPLEX_TYPE initialState[basisSize] = { { 1.0, 0.0 } };
	//the previous step vector
	COMPLEX_TYPE resultState[basisSize];

	Solver solver(0, 0.1, 10, model, NO_JUMP_RND_NUM_PROVIDER);
	solver.solve(output, initialState, resultState);

	Matcher<COMPLEX_TYPE> matchers[] = {
			//0
			ComplexEq8digits(COMPLEX_TYPE { 0.002677260879, -0.0002955831276 }),
			ComplexEq8digits(COMPLEX_TYPE { 0.1896210711, -0.02055884458 }),
			ComplexEq8digits(COMPLEX_TYPE { 0.1890809027, -0.0226279371 }),
			ComplexEq8digits(COMPLEX_TYPE { 0.06482264396, -0.009237247257 }),

			//4
			ComplexEq8digits(COMPLEX_TYPE { 0.1896210711, -0.02055884458 }),
			ComplexEq8digits(COMPLEX_TYPE { 0.3765648813, -0.04082210603 }),
			ComplexEq8digits(COMPLEX_TYPE { 0.376024713, -0.04289119856 }),
			ComplexEq8digits(COMPLEX_TYPE { 0.2517664542, -0.02950050871 }),

			//8
			ComplexEq8digits(COMPLEX_TYPE { 0.1890809027, -0.0226279371 }),
			ComplexEq8digits(COMPLEX_TYPE { 0.376024713, -0.04289119856 }),
			ComplexEq8digits(COMPLEX_TYPE { 0.3754845446, -0.04496029108 }),
			ComplexEq8digits(COMPLEX_TYPE { 0.2512262858, -0.03156960123 }),

			//12
			ComplexEq8digits(COMPLEX_TYPE { 0.06482264396, -0.009237247257 }),
			ComplexEq8digits(COMPLEX_TYPE { 0.2517664542, -0.02950050871 }),
			ComplexEq8digits(COMPLEX_TYPE { 0.2512262858, -0.03156960123 }),
			ComplexEq8digits(COMPLEX_TYPE { 0.126968027, -0.01817891139 })
	//
			};

	std::vector<COMPLEX_TYPE> cs(basisSize);
	cs.assign(resultState, resultState + basisSize);
	ASSERT_THAT(cs, ElementsAreArray(matchers))/*<< output.str()*/;
}

/**
 *	KAPPA = 1.0
 *	DELTA_OMEGA = 20.0
 *	G = 50.0
 *	LATIN_E = 2.0
 *	J = 0.1
 *
 *	atom1SSize = atom2SSize = 2
 *	field1SSize = field2SSize = 2
 *
 *	timeStep = 0.0001
 *	timeStepsNumber = 100000
 *
 *	No jumps
 *
 */
TEST (Solver, tenThousandStepsNoJump) {
	std::ostringstream output;

	Model model(2, 2, 2, 2, 1.0, 20.0, 50.0, 2.0, 0.1);

	const int basisSize = 16;

	//the ground state
	COMPLEX_TYPE initialState[basisSize] = { { 1.0, 0.0 } };
	//the previous step vector
	COMPLEX_TYPE resultState[basisSize];

	Solver solver(0, 0.0001, 100000, model, NO_JUMP_RND_NUM_PROVIDER);
	solver.solve(output, initialState, resultState);

	Matcher<COMPLEX_TYPE> matchers[] = {
			//0
			ComplexEq8digits(COMPLEX_TYPE { 0.9594776531, -0.183561852 }),
			ComplexEq8digits(COMPLEX_TYPE { -0.02060958666, -0.07750355967 }),
			ComplexEq8digits(COMPLEX_TYPE { -0.01686120876, -0.0792847911 }),
			ComplexEq8digits(COMPLEX_TYPE { -0.01789810978, -0.07879105994 }),

			//4
			ComplexEq8digits(COMPLEX_TYPE { -0.02060958666, -0.07750355967 }),
			ComplexEq8digits(COMPLEX_TYPE { 0.00114480127, 0.02855473262 }),
			ComplexEq8digits(COMPLEX_TYPE { 0.004893179177, 0.02677350119 }),
			ComplexEq8digits(COMPLEX_TYPE { 0.003856278156, 0.02726723235 }),

			//8
			ComplexEq8digits(COMPLEX_TYPE { -0.01686120876, -0.0792847911 }),
			ComplexEq8digits(COMPLEX_TYPE { 0.004893179177, 0.02677350119 }),
			ComplexEq8digits(COMPLEX_TYPE { 0.008641557085, 0.02499226976 }),
			ComplexEq8digits(COMPLEX_TYPE { 0.007604656063, 0.02548600092 }),

			//12
			ComplexEq8digits(COMPLEX_TYPE { -0.01789810978, -0.07879105994 }),
			ComplexEq8digits(COMPLEX_TYPE { 0.003856278156, 0.02726723235 }),
			ComplexEq8digits(COMPLEX_TYPE { 0.007604656063, 0.02548600092 }),
			ComplexEq8digits(COMPLEX_TYPE { 0.006567755041, 0.02597973208 })
	//
			};

	std::vector<COMPLEX_TYPE> cs(basisSize);
	cs.assign(resultState, resultState + basisSize);
	ASSERT_THAT(cs, ElementsAreArray(matchers))/*<< output.str()*/;
}

/**
 *	KAPPA = 1.0
 *	DELTA_OMEGA = 20.0
 *	G = 50.0
 *	LATIN_E = 2.0
 *	J = 0.1
 *
 *	atom1SSize = atom2SSize = 2
 *	field1SSize = field2SSize = 3
 *
 *	timeStep = 0.0001
 *	timeStepsNumber = 100000
 *
 *	No jumps
 *
 */
TEST (Solver, bigBasisNoJump) {
	std::ostringstream output;

	Model model(2, 2, 3, 3, 1.0, 20.0, 50.0, 2.0, 0.1);

	const int basisSize = 36;

	//the ground state
	COMPLEX_TYPE initialState[basisSize] = { { 1.0, 0.0 } };
	//the previous step vector
	COMPLEX_TYPE resultState[basisSize];

	Solver solver(0, 0.0001, 100000, model, NO_JUMP_RND_NUM_PROVIDER);
	solver.solve(output, initialState, resultState);

	Matcher<COMPLEX_TYPE> matchers[] = {
			//0
			ComplexEq8digits(COMPLEX_TYPE { 0.9740989849, -0.1229044446 }),
			ComplexEq8digits(COMPLEX_TYPE { -0.01346282435, -0.05535192581 }),
			ComplexEq8digits(COMPLEX_TYPE { -0.01178110752, -0.05611246831 }),
			ComplexEq8digits(COMPLEX_TYPE { -0.01227054271, -0.05589044469 }),

			//4
			ComplexEq8digits(COMPLEX_TYPE { -0.01225074642, -0.05590061811 }),
			ComplexEq8digits(COMPLEX_TYPE { -0.01226139305, -0.05589526458 }),
			ComplexEq8digits(COMPLEX_TYPE { -0.01346282435, -0.05535192581 }),
			ComplexEq8digits(COMPLEX_TYPE { 0.000193372048, 0.01220059298 }),

			//8
			ComplexEq8digits(COMPLEX_TYPE { 0.001875088875, 0.01144005048 }),
			ComplexEq8digits(COMPLEX_TYPE { 0.001385653693, 0.0116620741 }),
			ComplexEq8digits(COMPLEX_TYPE { 0.001405449979, 0.01165190067 }),
			ComplexEq8digits(COMPLEX_TYPE { 0.00139480335, 0.01165725421 }),

			//12
			ComplexEq8digits(COMPLEX_TYPE { -0.01178110752, -0.05611246831 }),
			ComplexEq8digits(COMPLEX_TYPE { 0.001875088875, 0.01144005048 }),
			ComplexEq8digits(COMPLEX_TYPE { 0.003556805702, 0.01067950797 }),
			ComplexEq8digits(COMPLEX_TYPE { 0.00306737052, 0.0109015316 }),

			//16
			ComplexEq8digits(COMPLEX_TYPE { 0.003087166806, 0.01089135817 }),
			ComplexEq8digits(COMPLEX_TYPE { 0.003076520177, 0.0108967117 }),
			ComplexEq8digits(COMPLEX_TYPE { -0.01227054271, -0.05589044469 }),
			ComplexEq8digits(COMPLEX_TYPE { 0.001385653693, 0.0116620741 }),

			//20
			ComplexEq8digits(COMPLEX_TYPE { 0.00306737052, 0.0109015316 }),
			ComplexEq8digits(COMPLEX_TYPE { 0.002577935338, 0.01112355523 }),
			ComplexEq8digits(COMPLEX_TYPE { 0.002597731624, 0.0111133818 }),
			ComplexEq8digits(COMPLEX_TYPE { 0.002587084995, 0.01111873533 }),

			//24
			ComplexEq8digits(COMPLEX_TYPE { -0.01225074642, -0.05590061811 }),
			ComplexEq8digits(COMPLEX_TYPE { 0.001405449979, 0.01165190067 }),
			ComplexEq8digits(COMPLEX_TYPE { 0.003087166806, 0.01089135817 }),
			ComplexEq8digits(COMPLEX_TYPE { 0.002597731624, 0.0111133818 }),

			//28
			ComplexEq8digits(COMPLEX_TYPE { 0.00261752791, 0.01110320837 }),
			ComplexEq8digits(COMPLEX_TYPE { 0.002606881281, 0.0111085619 }),
			ComplexEq8digits(COMPLEX_TYPE { -0.01226139305, -0.05589526458 }),
			ComplexEq8digits(COMPLEX_TYPE { 0.00139480335, 0.01165725421 }),

			//32
			ComplexEq8digits(COMPLEX_TYPE { 0.003076520177, 0.0108967117 }),
			ComplexEq8digits(COMPLEX_TYPE { 0.002587084995, 0.01111873533 }),
			ComplexEq8digits(COMPLEX_TYPE { 0.002606881281, 0.0111085619 }),
			ComplexEq8digits(COMPLEX_TYPE { 0.002596234652, 0.01111391543 })
	//
			};

	std::vector<COMPLEX_TYPE> cs(basisSize);
	cs.assign(resultState, resultState + basisSize);
	ASSERT_THAT(cs, ElementsAreArray(matchers))/*<< output.str()*/;
}

/**
 *	KAPPA = 1.0
 *	DELTA_OMEGA = 20.0
 *	G = 50.0
 *	LATIN_E = 2.0
 *	J = 0.1
 *
 *	atom1SSize = atom2SSize = 2
 *	field1SSize = field2SSize = 2
 *
 *	timeStep = 0.0001
 *	timeStepsNumber = 100000
 *
 */
TEST (Solver, makeJump) {
	std::ostringstream output;
	output << std::setprecision(12);

	Model model(2, 2, 2, 2, 1.0, 20.0, 50.0, 30.0, 0.1);

	const int basisSize = 16;

	COMPLEX_TYPE stateBeforeJump[basisSize];
	for (int i = 0; i < basisSize; ++i) {
		stateBeforeJump[i]= {1.0-1.0/basisSize*i};
	}
	COMPLEX_TYPE stateAfterJump[basisSize];

	class MockRndNumProvider: public RndNumProvider {
	public:
		void initBuffer(int streamId, FLOAT_TYPE *buffer, int bufferSize)
				override {
			int i = 0;
			buffer[i++] = 0.1; // the first cavity wins
			buffer[i++] = 0.7; // the second cavity wins
		}
	} rndNumProviderMock;

	Solver solver(0, 0.0001, 10000, model, rndNumProviderMock);

	//jumps in the first cavity
	solver.makeJump(output, stateBeforeJump, stateAfterJump);

	Matcher<COMPLEX_TYPE> matchersJFirstCavity[] = {
			//0
			ComplexEq8digits(COMPLEX_TYPE { 0.4666728031015298, 0.0 }),
			ComplexEq8digits(COMPLEX_TYPE { 0.4666728031, 0.0 }),
			ComplexEq8digits(COMPLEX_TYPE { 0.4666728031, 0.0 }),
			ComplexEq8digits(COMPLEX_TYPE { 0.4666728031015298, 0.0 }),

			//4
			ComplexEq8digits(COMPLEX_TYPE { 0.1794895397, 0.0 }),
			ComplexEq8digits(COMPLEX_TYPE { 0.1794895397, 0.0 }),
			ComplexEq8digits(COMPLEX_TYPE { 0.1794895397, 0.0 }),
			ComplexEq8digits(COMPLEX_TYPE { 0.17948953965443454, 0.0 }),

			//8
			ComplexEq8digits(COMPLEX_TYPE { 0.0, 0.0 }), //
			ComplexEq8digits(COMPLEX_TYPE { 0.0, 0.0 }), //
			ComplexEq8digits(COMPLEX_TYPE { 0.0, 0.0 }), //
			ComplexEq8digits(COMPLEX_TYPE { 0.0, 0.0 }), //

			//12
			ComplexEq8digits(COMPLEX_TYPE { 0.0, 0.0 }), //
			ComplexEq8digits(COMPLEX_TYPE { 0.0, 0.0 }), //
			ComplexEq8digits(COMPLEX_TYPE { 0.0, 0.0 }), //
			ComplexEq8digits(COMPLEX_TYPE { 0.0, 0.0 })
	//
			};

	std::vector<COMPLEX_TYPE> cs(basisSize);
	cs.assign(stateAfterJump, stateAfterJump + basisSize);
	ASSERT_THAT(cs, ElementsAreArray(matchersJFirstCavity));

	//jumps in the second cavity
	solver.makeJump(output, stateBeforeJump, stateAfterJump);

	COMPLEX_TYPE expectedSAfterSecondCJump[] = { { 0.3762883474, 0 }, {
			0.3292523039, 0 }, { 0., 0 }, { 0., 0 }, { 0.3762883474, 0 }, {
			0.3292523039, 0 }, { 0., 0 }, { 0., 0 }, { 0.3762883474, 0 }, {
			0.3292523039, 0 }, { 0., 0 }, { 0., 0 }, { 0.3762883474, 0 }, {
			0.3292523039, 0 }, { 0., 0 }, { 0., 0 } };

	std::cout << output.str();

	//check the matrix
	for (int i = 0; i < basisSize; ++i) {
		ASSERT_THAT(stateAfterJump[i],
				EqArrayComplexElementAt(expectedSAfterSecondCJump, i, 8));
	}

}

/**
 *	KAPPA = 1.0
 *	DELTA_OMEGA = 20.0
 *	G = 50.0
 *	LATIN_E = 2.0
 *	J = 0.1
 *
 *	atom1SSize = atom2SSize = 2
 *	field1SSize = field2SSize = 2
 *
 *	timeStep = 0.0001
 *	timeStepsNumber = 100000
 *
 *	Makes several jumps
 *
 */
TEST (Solver, severalJumps_simpleBasis) {
	std::ostringstream output;
	output << std::setprecision(12);

	Model model(2, 2, 2, 2, 1.0, 20.0, 50.0, 30.0, 0.1);

	const int basisSize = 16;

	class MockRndNumProvider: public RndNumProvider {
	public:
		void initBuffer(int streamId, FLOAT_TYPE *buffer, int bufferSize)
				override {
			int i = 0;
			buffer[i++] = 0.99; // the first jump
			buffer[i++] = 0.7; // used in the which cavity decision - the second cavity wins
			buffer[i++] = 0.98; // the second jump
			buffer[i++] = 0.2; // used in the which cavity decision - the first cavity wins
			buffer[i++] = 0.99; // the third jump
			buffer[i++] = 0.1; // used in the which cavity decision - the first cavity wins
			buffer[i++] = 0.0; // next threshold (is not possible to pass)
		}
	} rndNumProviderMock;

	//the ground state
	COMPLEX_TYPE initialState[basisSize] = { { 1.0, 0.0 } };
	//the previous step vector
	COMPLEX_TYPE resultState[basisSize];

	Solver solver(0, 0.0001, 10000, model, rndNumProviderMock);
	solver.solve(output, initialState, resultState);

	COMPLEX_TYPE expectedResultState[] = { { 0.6077136389, -0.2221199555 }, {
			0.2360838469, -0.0654836083 }, { 0.3119165305, -0.09798796695 }, {
			0.3071529809, -0.09182719887 }, { 0.1726922487, -0.08270619551 }, {
			-0.1989375433, 0.07393015168 }, { -0.1231048597, 0.04142579303 }, {
			-0.1278684093, 0.04758656112 }, { 0.2595078961, -0.1117267537 }, {
			-0.1121218959, 0.0449095935 }, { -0.03628921227, 0.01240523486 }, {
			-0.04105276187, 0.01856600294 }, { 0.2436579396, -0.1092507441 }, {
			-0.1279718524, 0.04738560305 }, { -0.05213916877, 0.0148812444 }, {
			-0.05690271837, 0.02104201248 } };

	std::cout << output.str();

	//check the matrix
	for (int i = 0; i < basisSize; ++i) {
		ASSERT_THAT(resultState[i],
				EqArrayComplexElementAt(expectedResultState, i, 8));
	}
}

/**
 *	KAPPA = 1.0
 *	DELTA_OMEGA = 20.0
 *	G = 50.0
 *	LATIN_E = 30.0
 *	J = 0.1
 *
 *	atom1SSize = atom2SSize = 2
 *	field1SSize = field2SSize = 4
 *
 *	timeStep = 0.0001
 *	timeStepsNumber = 100000
 *
 *	Makes several jumps
 *
 */
TEST (Solver, severalJumps_complexBasis) {
	std::ostringstream output;
	output << std::setprecision(12);

	Model model(2, 2, 4, 4, 1.0, 20.0, 50.0, 30.0, 0.1);

	const int basisSize = 64;

	class MockRndNumProvider: public RndNumProvider {
	public:
		void initBuffer(int streamId, FLOAT_TYPE *buffer, int bufferSize)
				override {
			int i = 0;
			buffer[i++] = 0.99; // the first jump
			buffer[i++] = 0.7; // used in the which cavity decision - the second cavity wins
			buffer[i++] = 0.98; // the second jump
			buffer[i++] = 0.2; // used in the which cavity decision - the first cavity wins
			buffer[i++] = 0.99; // the third jump
			buffer[i++] = 0.1; // used in the which cavity decision - the first cavity wins
			buffer[i++] = 0.0; // next threshold (is not possible to pass)
		}
	} rndNumProviderMock;

	//the ground state
	COMPLEX_TYPE initialState[basisSize] = { { 1.0, 0.0 } };
	//the previous step vector
	COMPLEX_TYPE resultState[basisSize];

	Solver solver(0, 0.0001, 10000, model, rndNumProviderMock);
	solver.solve(output, initialState, resultState);

	COMPLEX_TYPE expectedResultState[] = { { 0.2377753206, -0.4189228581 }, {
			0.100159848, -0.1759928413 }, { 0.1159140452, -0.2044619784 }, {
			0.1104845168, -0.1943438109 }, { 0.1124874778, -0.1984935242 }, {
			0.1112106118, -0.1957932676 }, { 0.1115667585, -0.196740822 }, {
			0.1114060556, -0.1963122955 }, { 0.1001574921, -0.175989115 }, {
			-0.03745798057, 0.06694090178 }, { -0.02170378332, 0.03847176472 },
			{ -0.02713331172, 0.04858993219 },
			{ -0.02513035072, 0.04444021891 },
			{ -0.02640721676, 0.04714047555 },
			{ -0.02605107008, 0.04619292108 },
			{ -0.02621177299, 0.04662144767 }, { 0.1159118648, -0.2044584962 },
			{ -0.02170360783, 0.03847152057 },
			{ -0.005949410576, 0.0100023835 },
			{ -0.01137893897, 0.02012055097 },
			{ -0.009375977978, 0.0159708377 },
			{ -0.01065284402, 0.01867109434 },
			{ -0.01029669734, 0.01772353987 },
			{ -0.01045740024, 0.01815206646 }, { 0.1104824642, -0.1943405702 },
			{ -0.02713300839, 0.04858944655 },
			{ -0.01137881114, 0.02012030948 },
			{ -0.01680833954, 0.03023847695 },
			{ -0.01480537854, 0.02608876368 },
			{ -0.01608224459, 0.02878902032 }, { -0.0157260979, 0.02784146585 },
			{ -0.01588680081, 0.02826999244 }, { 0.112485439, -0.1984902836 }, {
					-0.02513003365, 0.04443973323 }, { -0.009375836399,
					0.01597059616 }, { -0.0148053648, 0.02608876363 }, {
					-0.0128024038, 0.02193905036 }, { -0.01407926984,
					0.024639307 }, { -0.01372312316, 0.02369175253 }, {
					-0.01388382607, 0.02412027912 }, { 0.1112085876,
					-0.1957900679 }, { -0.02640688507, 0.04713994887 }, {
					-0.01065268781, 0.01867081181 }, { -0.01608221621,
					0.02878897928 }, { -0.01407925522, 0.024639266 }, {
					-0.01535612126, 0.02733952264 }, { -0.01499997458,
					0.02639196818 }, { -0.01516067748, 0.02682049476 }, {
					0.1115647867, -0.1967376937 }, { -0.02605068595,
					0.0461923231 }, { -0.0102964887, 0.01772318604 }, {
					-0.0157260171, 0.0278413535 }, { -0.0137230561,
					0.02369164023 }, { -0.01499992214, 0.02639189687 }, {
					-0.01464377546, 0.0254443424 }, { -0.01480447837,
					0.02587286899 }, { 0.1114038857, -0.1963088873 }, {
					-0.02621158697, 0.04662112953 }, { -0.01045738972,
					0.01815199247 }, { -0.01588691812, 0.02827015994 }, {
					-0.01388395712, 0.02412044666 }, { -0.01516082316,
					0.0268207033 }, { -0.01480467648, 0.02587314884 }, {
					-0.01496537939, 0.02630167542 } };

	std::cout << output.str();

	//check the matrix
	for (int i = 0; i < basisSize; ++i) {
		ASSERT_THAT(resultState[i],
				EqArrayComplexElementAt(expectedResultState, i, 8));
	}
}

