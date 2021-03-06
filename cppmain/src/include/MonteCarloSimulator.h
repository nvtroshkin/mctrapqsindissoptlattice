/*
 * MonteCarloSimulator.h
 *
 *  Created on: Nov 29, 2017
 *      Author: fakesci
 */

#ifndef SRC_INCLUDE_MONTECARLOSIMULATOR_H_
#define SRC_INCLUDE_MONTECARLOSIMULATOR_H_

#include <precision-definition.h>
#include <Model.h>
#include <SimulationResult.h>

class MonteCarloSimulator {
	const uint basisSize;
	const uint samplesNumber;

	Model &model;

	//used as initial state
	CUDA_COMPLEX_TYPE * groundState;

public:
	MonteCarloSimulator(uint samplesNumber, Model &model);

	~MonteCarloSimulator();

	SimulationResult *simulate(FLOAT_TYPE timeStep, uint nTimeSteps,
			uint threadsPerBlock, uint nBlocks);
};

#endif /* SRC_INCLUDE_MONTECARLOSIMULATOR_H_ */
