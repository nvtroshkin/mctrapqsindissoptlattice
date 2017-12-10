/*
 * constants.h
 *
 *  Created on: Nov 18, 2017
 *      Author: fakesci
 */

#ifndef SRC_EVAL_PARAMS_H_
#define SRC_EVAL_PARAMS_H_

#include <precision-definition.h>

/**
 * If defined, progress of calculations is printed every N samples
 */
#define PRINT_PROGRESS
static const int SAMPLES_BETWEEN_PROGRESS = 10;

/**
 * OpenMP threads to use
 */
#define THREADS_NUM 4

/**
 * Consider H as a sparse matrix
 * (don't know whether useful or not, may be for bigger dimensions)
 */
//#define H_SPARSE

/**
 * Print info about how dense matrices of the operators are
 */
#define CHECK_SPARSITY

//Evaluation of each sample is performed beginning at 0s and ending at the end time.
//Increasing the END_TIME value is necessary to caught the stationary evaluation
//phase
static const FLOAT_TYPE TIME_STEP_SIZE = 0.0001;
static const int TIME_STEPS_NUMBER = 100000;		//the total number of steps
static const int MONTE_CARLO_SAMPLES_NUMBER = 100;

static const FLOAT_TYPE EVAL_TIME = TIME_STEP_SIZE * TIME_STEPS_NUMBER;

//change to use another pseudo random sequence
static const int RANDSEED = 777;
/*
 * the size of the random numbers sequence available for each thread -
 * an exception is thrown if all numbers are used
 */
static const int RND_NUM_BUFF_SIZE = 128 * 1024;

//----------------debug info--------------------------------

/**
 * Defines how often debug info is printed
 */
static const int TIME_STEPS_BETWEEN_DEBUG = 1000;

/**
 * If defined, debug info relating to the continuous evolution is printed every
 * N steps of the solver
 */
//#define DEBUG_CONTINUOUS

/**
 * If defined, debug info related to random jumps is printed.
 * Some of it is printed when a jump occurred, some - every N steps
 */
//#define DEBUG_JUMPS

//----------------------------------------------------------

#endif /* SRC_EVAL_PARAMS_H_ */
