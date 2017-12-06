/*
 * constants.h
 *
 *  Created on: Nov 18, 2017
 *      Author: fakesci
 */

#ifndef SRC_EVAL_PARAMS_H_
#define SRC_EVAL_PARAMS_H_

#include <precision-definition.h>

//uncomment for extra output
//#define DEBUG_MODE
//#define DEBUG_JUMPS
#define PRINT_PROGRESS
static const int NOTIFY_EACH_N_SAMPLES = 10; //print info about progress each N processed samples

#define THREADS_NUM 4

//Evaluation of each sample is performed beginning at 0s and ending at the end time.
//Increasing the END_TIME value is necessary to caught the stationary evaluation
//phase
static const FLOAT_TYPE TIME_STEP_SIZE = 0.001;
static const int TIME_STEPS_NUMBER = 10000;		//the total number of steps by the time axis
static const int MONTE_CARLO_SAMPLES_NUMBER = 100;

static const FLOAT_TYPE EVAL_TIME = TIME_STEP_SIZE * TIME_STEPS_NUMBER;

//change to use another pseudo random sequence
static const int RANDSEED = 345777;
/*
 * the size of the random numbers sequence available for each thread -
 * an exception is thrown if all numbers are used
 */
static const int RND_NUM_BUFF_SIZE = 128 * 1024;

#endif /* SRC_EVAL_PARAMS_H_ */
