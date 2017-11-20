/*
 * constants.h
 *
 *  Created on: Nov 18, 2017
 *      Author: fake_sci
 */

#ifndef SRC_EVAL_PARAMS_H_
#define SRC_EVAL_PARAMS_H_

//the dressed basis macros
static const int MAX_PHOTON_NUMBER = 1;
static const int DRESSED_BASIS_SIZE = (2 * (MAX_PHOTON_NUMBER + 1));

//Evaluation of each sample is performed beginning at 0s and ending at the end time.
//Increasing the END_TIME value is necessary to caught the stationary evaluation
//phase
static const float EVAL_TIME = 10.0f;
static const int TIME_STEPS_NUMBER = 100;		//the total number of steps by the time axis
static const int MONTE_CARLO_SAMPLES_NUMBER = 100;

#endif /* SRC_EVAL_PARAMS_H_ */
