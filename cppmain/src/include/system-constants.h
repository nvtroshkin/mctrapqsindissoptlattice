/*
 * system-constants.h
 *
 *  Created on: Nov 18, 2017
 *      Author: fakesci
 */

#ifndef SRC_SYSTEM_CONSTANTS_H_
#define SRC_SYSTEM_CONSTANTS_H_

#include <precision-definition.h>
#include <mkl-constants.h>

/**
 * Match actual levels number of atoms
 */
static const MKL_INT ATOM_1_LEVELS_NUMBER = 2;
static const MKL_INT ATOM_2_LEVELS_NUMBER = 2;
static const MKL_INT ATOM_3_LEVELS_NUMBER = 2;

/**
 * Because there are infinite number of Fock states for a field,
 * this introduces levels where the states are cut (greatly affects precision
 * if is not enough high and performance if chosen excessively high)
 */
static const MKL_INT FIELD_1_FOCK_STATES_NUMBER = 2;
static const MKL_INT FIELD_2_FOCK_STATES_NUMBER = 2;
static const MKL_INT FIELD_3_FOCK_STATES_NUMBER = 2;

//subspaces sizes
static const uint CAVITY_1_SUBSPACE_SIZE = ATOM_1_LEVELS_NUMBER
		* FIELD_1_FOCK_STATES_NUMBER;
static const uint CAVITY_2_SUBSPACE_SIZE = ATOM_2_LEVELS_NUMBER
		* FIELD_2_FOCK_STATES_NUMBER;
static const uint CAVITY_3_SUBSPACE_SIZE = ATOM_3_LEVELS_NUMBER
		* FIELD_3_FOCK_STATES_NUMBER;

static const uint BASIS_SIZE = CAVITY_1_SUBSPACE_SIZE * CAVITY_2_SUBSPACE_SIZE
		* CAVITY_3_SUBSPACE_SIZE;

//Parameters of the system
static const FLOAT_TYPE KAPPA = 1.0;
static const FLOAT_TYPE DELTA_OMEGA = 20.0;
static const FLOAT_TYPE G = 50.0;
static const FLOAT_TYPE scE = 10.0;
static const FLOAT_TYPE J = 0.1;

#endif /* SRC_SYSTEM_CONSTANTS_H_ */
