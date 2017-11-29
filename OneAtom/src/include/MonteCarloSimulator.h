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

class MonteCarloSimulator {
	const MKL_INT basisSize;
	const MKL_INT samplesNumber;

	Solver &solver;

	// a vector with all zeros
	COMPLEX_TYPE * const zeroVector;
	//used as initial state
	COMPLEX_TYPE * const groundState;

public:
	MonteCarloSimulator(MKL_INT basisSize, MKL_INT samplesNumber,
			Solver &solver);
	~MonteCarloSimulator();

	SimulationResult *simulate(std::ostream &consoleStream);
};

#endif /* SRC_INCLUDE_MONTECARLOSIMULATOR_H_ */
