/*
 * model-tests.cpp
 *
 *  Created on: Dec 8, 2017
 *      Author: fakesci
 */

#include "definitions.h"
#include "Model.h"
#include "utilities.h"

/**
 * a1Plus=
 *
 * 0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
 * 0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
 * 0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
 * 0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
 * 0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
 * 0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
 * 0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
 * 0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
 * 1	1	1	1	0	0	0	0	0	0	0	0	0	0	0	0
 * 1	1	1	1	0	0	0	0	0	0	0	0	0	0	0	0
 * 1	1	1	1	0	0	0	0	0	0	0	0	0	0	0	0
 * 1	1	1	1	0	0	0	0	0	0	0	0	0	0	0	0
 * 0	0	0	0	1	1	1	1	0	0	0	0	0	0	0	0
 * 0	0	0	0	1	1	1	1	0	0	0	0	0	0	0	0
 * 0	0	0	0	1	1	1	1	0	0	0	0	0	0	0	0
 * 0	0	0	0	1	1	1	1	0	0	0	0	0	0	0	0
 *
 *
 */
TEST (Model, a1Plus_matrixElements) {
	const FLOAT_TYPE EXPECTED_RESULT[16][16] = {
	//0
			{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
			//
			{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
			//
			{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
			//
			{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },

			//4
			{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
			//
			{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
			//
			{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
			//
			{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },

			//8
			{ 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
			//
			{ 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
			//
			{ 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
			//
			{ 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },

			//12
			{ 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0 },
			//
			{ 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0 },
			//
			{ 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0 },
			//
			{ 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0 },
	//
			};

	Model model(2, 2, 2, 2, 1.0, 20.0, 50.0, 2.0, 0.1);

	//check the matrix
	for (int i = 0; i < 16; ++i) {
		for (int j = 0; j < 16; ++j) {
			ASSERT_THAT(model.a1Plus(i, j),
					EqMatrixElementAt(EXPECTED_RESULT, i, j, 8));
		}
	}
}

/**
 * a1=
 *
 * 0	0	0	0	0	0	0	0	1	1	1	1	0	0	0	0
 * 0	0	0	0	0	0	0	0	1	1	1	1	0	0	0	0
 * 0	0	0	0	0	0	0	0	1	1	1	1	0	0	0	0
 * 0	0	0	0	0	0	0	0	1	1	1	1	0	0	0	0
 * 0	0	0	0	0	0	0	0	0	0	0	0	1	1	1	1
 * 0	0	0	0	0	0	0	0	0	0	0	0	1	1	1	1
 * 0	0	0	0	0	0	0	0	0	0	0	0	1	1	1	1
 * 0	0	0	0	0	0	0	0	0	0	0	0	1	1	1	1
 * 0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
 * 0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
 * 0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
 * 0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
 * 0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
 * 0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
 * 0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
 * 0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
 *
 */
TEST (Model, a1_matrixElements) {
	const FLOAT_TYPE EXPECTED_RESULT[16][16] = {
	//0
			{ 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0 },
			//
			{ 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0 },
			//
			{ 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0 },
			//
			{ 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0 },

			//4
			{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1 },
			//
			{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1 },
			//
			{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1 },
			//
			{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1 },

			//8
			{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
			//
			{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
			//
			{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
			//
			{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },

			//12
			{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
			//
			{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
			//
			{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
			//
			{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }
	//
			};

	Model model(2, 2, 2, 2, 1.0, 20.0, 50.0, 2.0, 0.1);

	//check the matrix
	for (int i = 0; i < 16; ++i) {
		for (int j = 0; j < 16; ++j) {
			ASSERT_THAT(model.a1(i, j), EqMatrixElementAt(EXPECTED_RESULT, i, j, 8));
		}
	}
}

/**
 * sigma1Plus=
 *
 * 0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
 * 0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
 * 0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
 * 0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
 * 1	1	1	1	0	0	0	0	0	0	0	0	0	0	0	0
 * 1	1	1	1	0	0	0	0	0	0	0	0	0	0	0	0
 * 1	1	1	1	0	0	0	0	0	0	0	0	0	0	0	0
 * 1	1	1	1	0	0	0	0	0	0	0	0	0	0	0	0
 * 0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
 * 0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
 * 0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
 * 0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
 * 0	0	0	0	0	0	0	0	1	1	1	1	0	0	0	0
 * 0	0	0	0	0	0	0	0	1	1	1	1	0	0	0	0
 * 0	0	0	0	0	0	0	0	1	1	1	1	0	0	0	0
 * 0	0	0	0	0	0	0	0	1	1	1	1	0	0	0	0
 *
 */
TEST (Model, sigma1Plus_matrixElements) {
	const FLOAT_TYPE EXPECTED_RESULT[16][16] = {
	//0

			{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
			//
			{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
			//
			{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
			//
			{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },

			//4
			{ 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
			//
			{ 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
			//
			{ 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
			//
			{ 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },

			//8
			{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
			//
			{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
			//
			{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
			//
			{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },

			//12
			{ 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0 },
			//
			{ 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0 },
			//
			{ 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0 },
			//
			{ 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0 },
	//
			};

	Model model(2, 2, 2, 2, 1.0, 20.0, 50.0, 2.0, 0.1);

	//check the matrix
	for (int i = 0; i < 16; ++i) {
		for (int j = 0; j < 16; ++j) {
			ASSERT_THAT(model.sigma1Plus(i, j),
					EqMatrixElementAt(EXPECTED_RESULT, i, j, 8));
		}
	}
}

/**
 * sigma1Minus=
 *
 * 0	0	0	0	1	1	1	1	0	0	0	0	0	0	0	0
 * 0	0	0	0	1	1	1	1	0	0	0	0	0	0	0	0
 * 0	0	0	0	1	1	1	1	0	0	0	0	0	0	0	0
 * 0	0	0	0	1	1	1	1	0	0	0	0	0	0	0	0
 * 0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
 * 0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
 * 0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
 * 0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
 * 0	0	0	0	0	0	0	0	0	0	0	0	1	1	1	1
 * 0	0	0	0	0	0	0	0	0	0	0	0	1	1	1	1
 * 0	0	0	0	0	0	0	0	0	0	0	0	1	1	1	1
 * 0	0	0	0	0	0	0	0	0	0	0	0	1	1	1	1
 * 0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
 * 0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
 * 0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
 * 0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
 *
 */
TEST (Modle,sigma1Minus_matrixElements) {
	const FLOAT_TYPE EXPECTED_RESULT[16][16] = {
	//0
			{ 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0 },
			//
			{ 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0 },
			//
			{ 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0 },
			//
			{ 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0 },

			//4
			{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
			//
			{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
			//
			{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
			//
			{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },

			//8
			{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1 },
			//
			{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1 },
			//
			{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1 },
			//
			{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1 },

			//12
			{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
			//
			{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
			//
			{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
			//
			{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },

	//
			};

	Model model(2, 2, 2, 2, 1.0, 20.0, 50.0, 2.0, 0.1);

	//check the matrix
	for (int i = 0; i < 16; ++i) {
		for (int j = 0; j < 16; ++j) {
			ASSERT_THAT(model.sigma1Minus(i, j),
					EqMatrixElementAt(EXPECTED_RESULT, i, j, 8));
		}
	}
}

/**
 * a1Plus=
 *
 * 0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
 * 0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
 * 1	0	0	0	1	0	0	0	1	0	0	0	1	0	0	0
 * 0	1	0	0	0	1	0	0	0	1	0	0	0	1	0	0
 * 0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
 * 0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
 * 1	0	0	0	1	0	0	0	1	0	0	0	1	0	0	0
 * 0	1	0	0	0	1	0	0	0	1	0	0	0	1	0	0
 * 0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
 * 0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
 * 1	0	0	0	1	0	0	0	1	0	0	0	1	0	0	0
 * 0	1	0	0	0	1	0	0	0	1	0	0	0	1	0	0
 * 0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
 * 0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
 * 1	0	0	0	1	0	0	0	1	0	0	0	1	0	0	0
 * 0	1	0	0	0	1	0	0	0	1	0	0	0	1	0	0
 *
 */
TEST (Model, a2Plus_matrixElements) {
	const FLOAT_TYPE EXPECTED_RESULT[16][16] = {
	//0
			{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
			//
			{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
			//
			{ 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0 },
			//
			{ 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0 },

			//4
			{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
			//
			{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
			//
			{ 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0 },
			//
			{ 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0 },

			//8
			{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
			//
			{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
			//
			{ 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0 },
			//
			{ 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0 },

			//12
			{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
			//
			{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
			//
			{ 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0 },
			//
			{ 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0 },
	//
			};

	Model model(2, 2, 2, 2, 1.0, 20.0, 50.0, 2.0, 0.1);

	//check the matrix
	for (int i = 0; i < 16; ++i) {
		for (int j = 0; j < 16; ++j) {
			ASSERT_THAT(model.a2Plus(i, j),
					EqMatrixElementAt(EXPECTED_RESULT, i, j, 8));
		}
	}
}

/**
 * a2=
 *
 * 0	0	1	0	0	0	1	0	0	0	1	0	0	0	1	0
 * 0	0	0	1	0	0	0	1	0	0	0	1	0	0	0	1
 * 0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
 * 0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
 * 0	0	1	0	0	0	1	0	0	0	1	0	0	0	1	0
 * 0	0	0	1	0	0	0	1	0	0	0	1	0	0	0	1
 * 0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
 * 0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
 * 0	0	1	0	0	0	1	0	0	0	1	0	0	0	1	0
 * 0	0	0	1	0	0	0	1	0	0	0	1	0	0	0	1
 * 0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
 * 0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
 * 0	0	1	0	0	0	1	0	0	0	1	0	0	0	1	0
 * 0	0	0	1	0	0	0	1	0	0	0	1	0	0	0	1
 * 0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
 * 0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
 *
 */
TEST (Model, a2_matrixElements) {
	const FLOAT_TYPE EXPECTED_RESULT[16][16] = {
	//0
			{ 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0 },
			//
			{ 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1 },
			//
			{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
			//
			{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },

			//4
			{ 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0 },
			//
			{ 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1 },
			//
			{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
			//
			{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },

			//8
			{ 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0 },
			//
			{ 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1 },
			//
			{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
			//
			{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },

			//12
			{ 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0 },
			//
			{ 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1 },
			//
			{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
			//
			{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
	//
			};

	Model model(2, 2, 2, 2, 1.0, 20.0, 50.0, 2.0, 0.1);

	//check the matrix
	for (int i = 0; i < 16; ++i) {
		for (int j = 0; j < 16; ++j) {
			ASSERT_THAT(model.a2(i, j), EqMatrixElementAt(EXPECTED_RESULT, i, j, 8));
		}
	}
}

/**
 * sigma2Plus=
 *
 * 0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
 * 1	0	0	0	1	0	0	0	1	0	0	0	1	0	0	0
 * 0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
 * 0	0	1	0	0	0	1	0	0	0	1	0	0	0	1	0
 * 0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
 * 1	0	0	0	1	0	0	0	1	0	0	0	1	0	0	0
 * 0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
 * 0	0	1	0	0	0	1	0	0	0	1	0	0	0	1	0
 * 0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
 * 1	0	0	0	1	0	0	0	1	0	0	0	1	0	0	0
 * 0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
 * 0	0	1	0	0	0	1	0	0	0	1	0	0	0	1	0
 * 0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
 * 1	0	0	0	1	0	0	0	1	0	0	0	1	0	0	0
 * 0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
 * 0	0	1	0	0	0	1	0	0	0	1	0	0	0	1	0
 *
 */
TEST (Model, sigma2Plus_matrixElements) {
	const FLOAT_TYPE EXPECTED_RESULT[16][16] = {
	//0

			{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
			//
			{ 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0 },
			//
			{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
			//
			{ 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0 },

			//4
			{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
			//
			{ 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0 },
			//
			{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
			//
			{ 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0 },

			//8
			{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
			//
			{ 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0 },
			//
			{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
			//
			{ 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0 },

			//12
			{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
			//
			{ 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0 },
			//
			{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
			//
			{ 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0 },
	//
			};

	Model model(2, 2, 2, 2, 1.0, 20.0, 50.0, 2.0, 0.1);

	//check the matrix
	for (int i = 0; i < 16; ++i) {
		for (int j = 0; j < 16; ++j) {
			ASSERT_THAT(model.sigma2Plus(i, j),
					EqMatrixElementAt(EXPECTED_RESULT, i, j, 8));
		}
	}
}

/**
 * sigma2Minus=
 *
 * 0	1	0	0	0	1	0	0	0	1	0	0	0	1	0	0
 * 0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
 * 0	0	0	1	0	0	0	1	0	0	0	1	0	0	0	1
 * 0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
 * 0	1	0	0	0	1	0	0	0	1	0	0	0	1	0	0
 * 0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
 * 0	0	0	1	0	0	0	1	0	0	0	1	0	0	0	1
 * 0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
 * 0	1	0	0	0	1	0	0	0	1	0	0	0	1	0	0
 * 0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
 * 0	0	0	1	0	0	0	1	0	0	0	1	0	0	0	1
 * 0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
 * 0	1	0	0	0	1	0	0	0	1	0	0	0	1	0	0
 * 0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
 * 0	0	0	1	0	0	0	1	0	0	0	1	0	0	0	1
 * 0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0
 *
 */
TEST (Model, sigma2Minus_matrixElements) {
	const FLOAT_TYPE EXPECTED_RESULT[16][16] = {
	//0
			{ 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0 },
			//
			{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
			//
			{ 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1 },
			//
			{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },

			//4
			{ 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0 },
			//
			{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
			//
			{ 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1 },
			//
			{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },

			//8
			{ 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0 },
			//
			{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
			//
			{ 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1 },
			//
			{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },

			//12
			{ 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0 },
			//
			{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
			//
			{ 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1 },
			//
			{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },

	//
			};

	Model model(2, 2, 2, 2, 1.0, 20.0, 50.0, 2.0, 0.1);

	//check the matrix
	for (int i = 0; i < 16; ++i) {
		for (int j = 0; j < 16; ++j) {
			ASSERT_THAT(model.sigma2Minus(i, j),
					EqMatrixElementAt(EXPECTED_RESULT, i, j, 8));
		}
	}
}

/**
 * L =
 *
 * 0. +0. ,	0. +0. I	0. -2. I	0. +0. I	0. +0. I	0. +0. I	0. -2. I	0. +0. I	0. -2. I	0. -2. I	0. -4. I	0. -2. I	0. +0. I	0. +0. I	0. -2. I	0. +0. I
 * 0. +0. I	0. -80. I	0. -200. I	0. -2. I	0. +0. I	0. -80. I	0. -200. I	0. -2. I	0. -2. I	0. -82. I	0. -202. I	0. -4. I	0. +0. I	0. -80. I	0. -200. I	0. -2. I
 * 0. -2. I	0. -200. I	-4.-80. I	0. +0. I	0. -2. I	0. -200. I	-4.-80. I	0. +0. I	0. -3.9 I	0. -201.9 I	-4.-81.9 I	0. -1.9 I	0. -1.9 I	0. -199.9 I	-4.-79.9 I	0. +0.1 I
 * 0. +0. I	0. -2. I	0. +0. I	-4.-160. I	0. +0. I	0. -2. I	0. +0. I	-4.-160. I	0. -1.9 I	0. -3.9 I	0. -1.9 I	-4.-161.9 I	0. +0.1 I	0. -1.9 I	0. +0.1 I	-4.-159.9 I
 * 0. +0. I	0. +0. I	0. -2. I	0. +0. I	0. -80. I	0. -80. I	0. -82. I	0. -80. I	0. -200. I	0. -200. I	0. -202. I	0. -200. I	0. -2. I	0. -2. I	0. -4. I	0. -2. I
 * 0. +0. I	0. -80. I	0. -200. I	0. -2. I	0. -80. I	0. -160. I	0. -280. I	0. -82. I	0. -200. I	0. -280. I	0. -400. I	0. -202. I	0. -2. I	0. -82. I	0. -202. I	0. -4. I
 * 0. -2. I	0. -200. I	-4.-80. I	0. +0. I	0. -82. I	0. -280. I	-4.-160. I	0. -80. I	0. -201.9 I	0. -399.9 I	-4.-279.9 I	0. -199.9 I	0. -3.9 I	0. -201.9 I	-4.-81.9 I	0. -1.9 I
 * 0. +0. I	0. -2. I	0. +0. I	-4.-160. I	0. -80. I	0. -82. I	0. -80. I	-4.-240. I	0. -199.9 I	0. -201.9 I	0. -199.9 I	-4.-359.9 I	0. -1.9 I	0. -3.9 I	0. -1.9 I	-4.-161.9 I
 * 0. -2. I	0. -2. I	0. -3.9 I	0. -1.9 I	0. -200. I	0. -200. I	0. -201.9 I	0. -199.9 I	-4.-80. I	-4.-80. I	-4.-81.9 I	-4.-79.9 I	0. +0. I	0. +0. I	0. -1.9 I	0. +0.1 I
 * 0. -2. I	0. -82. I	0. -201.9 I	0. -3.9 I	0. -200. I	0. -280. I	0. -399.9 I	0. -201.9 I	-4.-80. I	-4.-160. I	-4.-279.9 I	-4.-81.9 I	0. +0. I	0. -80. I	0. -199.9 I	0. -1.9 I
 * 0. -4. I	0. -202. I	-4.-81.9 I	0. -1.9 I	0. -202. I	0. -400. I	-4.-279.9 I	0. -199.9 I	-4.-81.9 I	-4.-279.9 I	-8.-159.8 I	-4.-79.8 I	0. -1.9 I	0. -199.9 I	-4.-79.8 I	0. +0.2 I
 * 0. -2. I	0. -4. I	0. -1.9 I	-4.-161.9 I	0. -200. I	0. -202. I	0. -199.9 I	-4.-359.9 I	-4.-79.9 I	-4.-81.9 I	-4.-79.8 I	-8.-239.8 I	0. +0.1 I	0. -1.9 I	0. +0.2 I	-4.-159.8 I
 * 0. +0. I	0. +0. I	0. -1.9 I	0. +0.1 I	0. -2. I	0. -2. I	0. -3.9 I	0. -1.9 I	0. +0. I	0. +0. I	0. -1.9 I	0. +0.1 I	-4.-160. I	-4.-160. I	-4.-161.9 I	-4.-159.9 I
 * 0. +0. I	0. -80. I	0. -199.9 I	0. -1.9 I	0. -2. I	0. -82. I	0. -201.9 I	0. -3.9 I	0. +0. I	0. -80. I	0. -199.9 I	0. -1.9 I	-4.-160. I	-4.-240. I	-4.-359.9 I	-4.-161.9 I
 * 0. -2. I	0. -200. I	-4.-79.9 I	0. +0.1 I	0. -4. I	0. -202. I	-4.-81.9 I	0. -1.9 I	0. -1.9 I	0. -199.9 I	-4.-79.8 I	0. +0.2 I	-4.-161.9 I	-4.-359.9 I	-8.-239.8 I	-4.-159.8 I
 * 0. +0. I	0. -2. I	0. +0.1 I	-4.-159.9 I	0. -2. I	0. -4. I	0. -1.9 I	-4.-161.9 I	0. +0.1 I	0. -1.9 I	0. +0.2 I	-4.-159.8 I	-4.-159.9 I	-4.-161.9 I	-4.-159.8 I	-8.-319.8 I
 *
 *
 */
TEST (Model, L_matrixElements) {
	const COMPLEX_TYPE EXPECTED_RESULT[16][16] = { //
			//0
					{ { 0.0, 0.0 }, { 0.0, 0.0 }, { 0.0, -2.0 }, { 0.0, 0.0 }, {
							0.0, 0.0 }, { 0.0, 0.0 }, { 0.0, -2.0 },
							{ 0.0, 0.0 }, { 0.0, -2.0 }, { 0.0, -2.0 }, { 0.0,
									-4.0 }, { 0.0, -2.0 }, { 0.0, +0.0 }, { 0.0,
									+0.0 }, { 0.0, -2.0 }, { 0.0, +0.0 } },
					//1
					{ { 0.0, +0.0 }, { 0.0, -80.0 }, { 0.0, -200.0 }, { 0.0,
							-2.0 }, { 0.0, +0.0 }, { 0.0, -80.0 },
							{ 0.0, -200.0 }, { 0.0, -2.0 }, { 0.0, -2.0 }, {
									0.0, -82.0 }, { 0.0, -202.0 },
							{ 0.0, -4.0 }, { 0.0, +0.0 }, { 0.0, -80.0 }, { 0.0,
									-200.0 }, { 0.0, -2.0 } },
					//2
					{ { 0.0, -2.0 }, { 0.0, -200.0 }, { -4.0, -80.0 }, { 0.0,
							+0.0 }, { 0.0, -2.0 }, { 0.0, -200.0 }, { -4.0,
							-80.0 }, { 0.0, +0.0 }, { 0.0, -3.9 },
							{ 0.0, -201.9 }, { -4., -81.9 }, { 0.0, -1.9 }, {
									0.0, -1.9 }, { 0.0, -199.9 },
							{ -4., -79.9 }, { 0.0, +0.1 } },
					//3
					{ { 0.0, +0.0 }, { 0.0, -2.0 }, { 0.0, +0.0 },
							{ -4., -160.0 }, { 0.0, +0.0 }, { 0.0, -2.0 }, {
									0.0, +0.0 }, { -4., -160.0 }, { 0.0, -1.9 },
							{ 0.0, -3.9 }, { 0.0, -1.9 }, { -4., -161.9 }, {
									0.0, +0.1 }, { 0.0, -1.9 }, { 0.0, +0.1 }, {
									-4., -159.9 } },
					//4
					{ { 0.0, +0.0 }, { 0.0, +0.0 }, { 0.0, -2.0 },
							{ 0.0, +0.0 }, { 0.0, -80.0 }, { 0.0, -80.0 }, {
									0.0, -82.0 }, { 0.0, -80.0 },
							{ 0.0, -200.0 }, { 0.0, -200.0 }, { 0.0, -202.0 }, {
									0.0, -200.0 }, { 0.0, -2.0 }, { 0.0, -2.0 },
							{ 0.0, -4.0 }, { 0.0, -2.0 } },
					//5
					{ { 0.0, +0.0 }, { 0.0, -80.0 }, { 0.0, -200.0 }, { 0.0,
							-2.0 }, { 0.0, -80.0 }, { 0.0, -160.0 }, { 0.0,
							-280.0 }, { 0.0, -82.0 }, { 0.0, -200.0 }, { 0.0,
							-280.0 }, { 0.0, -400.0 }, { 0.0, -202.0 }, { 0.0,
							-2.0 }, { 0.0, -82.0 }, { 0.0, -202.0 },
							{ 0.0, -4.0 } },
					//6
					{ { 0.0, -2.0 }, { 0.0, -200.0 }, { -4., -80.0 }, { 0.0,
							+0.0 }, { 0.0, -82.0 }, { 0.0, -280.0 }, { -4.,
							-160.0 }, { 0.0, -80.0 }, { 0.0, -201.9 }, { 0.0,
							-399.9 }, { -4., -279.9 }, { 0.0, -199.9 }, { 0.0,
							-3.9 }, { 0.0, -201.9 }, { -4., -81.9 },
							{ 0.0, -1.9 } },
					//7
					{ { 0.0, +0.0 }, { 0.0, -2.0 }, { 0.0, +0.0 },
							{ -4., -160.0 }, { 0.0, -80.0 }, { 0.0, -82.0 }, {
									0.0, -80.0 }, { -4., -240.0 },
							{ 0.0, -199.9 }, { 0.0, -201.9 }, { 0.0, -199.9 }, {
									-4., -359.9 }, { 0.0, -1.9 }, { 0.0, -3.9 },
							{ 0.0, -1.9 }, { -4., -161.9 } },
					//8
					{ { 0.0, -2.0 }, { 0.0, -2.0 }, { 0.0, -3.9 },
							{ 0.0, -1.9 }, { 0.0, -200.0 }, { 0.0, -200.0 }, {
									0.0, -201.9 }, { 0.0, -199.9 },
							{ -4., -80.0 }, { -4., -80.0 }, { -4., -81.9 }, {
									-4., -79.9 }, { 0.0, +0.0 }, { 0.0, +0.0 },
							{ 0.0, -1.9 }, { 0.0, +0.1 } },
					//9
					{ { 0.0, -2.0 }, { 0.0, -82.0 }, { 0.0, -201.9 }, { 0.0,
							-3.9 }, { 0.0, -200.0 }, { 0.0, -280.0 }, { 0.0,
							-399.9 }, { 0.0, -201.9 }, { -4., -80.0 }, { -4.,
							-160.0 }, { -4., -279.9 }, { -4., -81.9 }, { 0.0,
							+0.0 }, { 0.0, -80.0 }, { 0.0, -199.9 },
							{ 0.0, -1.9 } },
					//10
					{ { 0.0, -4.0 }, { 0.0, -202.0 }, { -4., -81.9 }, { 0.0,
							-1.9 }, { 0.0, -202.0 }, { 0.0, -400.0 }, { -4.,
							-279.9 }, { 0.0, -199.9 }, { -4., -81.9 }, { -4.,
							-279.9 }, { -8., -159.8 }, { -4., -79.8 }, { 0.0,
							-1.9 }, { 0.0, -199.9 }, { -4., -79.8 },
							{ 0.0, +0.2 } },
					//11
					{ { 0.0, -2.0 }, { 0.0, -4.0 }, { 0.0, -1.9 },
							{ -4., -161.9 }, { 0.0, -200.0 }, { 0.0, -202.0 }, {
									0.0, -199.9 }, { -4., -359.9 },
							{ -4., -79.9 }, { -4., -81.9 }, { -4., -79.8 }, {
									-8., -239.8 }, { 0.0, +0.1 }, { 0.0, -1.9 },
							{ 0.0, +0.2 }, { -4., -159.8 } },
					//12
					{ { 0.0, +0.0 }, { 0.0, +0.0 }, { 0.0, -1.9 },
							{ 0.0, +0.1 }, { 0.0, -2.0 }, { 0.0, -2.0 }, { 0.0,
									-3.9 }, { 0.0, -1.9 }, { 0.0, +0.0 }, { 0.0,
									+0.0 }, { 0.0, -1.9 }, { 0.0, +0.1 }, { -4.,
									-160.0 }, { -4., -160.0 }, { -4., -161.9 },
							{ -4., -159.9 } },
					//13
					{ { 0.0, +0.0 }, { 0.0, -80.0 }, { 0.0, -199.9 }, { 0.0,
							-1.9 }, { 0.0, -2.0 }, { 0.0, -82.0 },
							{ 0.0, -201.9 }, { 0.0, -3.9 }, { 0.0, +0.0 }, {
									0.0, -80.0 }, { 0.0, -199.9 },
							{ 0.0, -1.9 }, { -4., -160.0 }, { -4., -240.0 }, {
									-4., -359.9 }, { -4., -161.9 } },
					//14
					{ { 0.0, -2.0 }, { 0.0, -200.0 }, { -4., -79.9 }, { 0.0,
							+0.1 }, { 0.0, -4.0 }, { 0.0, -202.0 }, { -4.0,
							-81.9 }, { 0.0, -1.9 }, { 0.0, -1.9 },
							{ 0.0, -199.9 }, { -4., -79.8 }, { 0.0, +0.2 }, {
									-4., -161.9 }, { -4., -359.9 }, { -8.,
									-239.8 }, { -4., -159.8 } },
					//15
					{ { 0.0, +0.0 }, { 0.0, -2.0 }, { 0.0, +0.1 },
							{ -4., -159.9 }, { 0.0, -2.0 }, { 0.0, -4.0 }, {
									0.0, -1.9 }, { -4., -161.9 }, { 0.0, +0.1 },
							{ 0.0, -1.9 }, { 0.0, +0.2 }, { -4., -159.8 }, {
									-4., -159.9 }, { -4., -161.9 }, { -4.,
									-159.8 }, { -8., -319.8 } },

			};

	Model model(2, 2, 2, 2, 1.0, 20.0, 50.0, 2.0, 0.1);

	//check the matrix
	for (int i = 0; i < 16; ++i) {
		for (int j = 0; j < 16; ++j) {
			ASSERT_THAT(model.L(i, j),
					EqMatrixComplexElementAt(EXPECTED_RESULT, i, j, 8));
		}
	}
}
