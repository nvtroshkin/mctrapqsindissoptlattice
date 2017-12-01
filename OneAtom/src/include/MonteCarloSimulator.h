/*
 * MonteCarloSimulator.h
 *
 *  Created on: Nov 29, 2017
 *      Author: fake_sci
 */

#ifndef SRC_INCLUDE_MONTECARLOSIMULATOR_H_
#define SRC_INCLUDE_MONTECARLOSIMULATOR_H_

#include <precision-definition.h>
#include <Solver.h>
#include <CSR3Matrix.h>
#include <SimulationResult.h>
#include <iostream>
#include <ModelBuilder.h>
#include <RndNumProvider.h>

class MonteCarloSimulator {
	const MKL_INT basisSize;
	const MKL_INT samplesNumber;
	const int nThreads;

	ModelBuilder &modelBuilder;
	RndNumProvider &rndNumProvider;

	// a vector with all zeros
	COMPLEX_TYPE * const zeroVector;
	//used as initial state
	COMPLEX_TYPE * const groundState;

public:
	MonteCarloSimulator(MKL_INT basisSize, MKL_INT samplesNumber, int nThreads,
			ModelBuilder &modelBuilder, RndNumProvider &rndNumProvider);
	~MonteCarloSimulator();

	SimulationResult *simulate(std::ostream &consoleStream, FLOAT_TYPE timeStep,
			int nTimeSteps);
};

#endif /* SRC_INCLUDE_MONTECARLOSIMULATOR_H_ */
