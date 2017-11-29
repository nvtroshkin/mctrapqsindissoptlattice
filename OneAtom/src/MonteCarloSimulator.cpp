/*
 * MonteCarloSimulator.cpp
 *
 *  Created on: Nov 29, 2017
 *      Author: fake_sci
 */

#include <mkl-constants.h>
#include <precision-definition.h>
#include <Solver.h>
#include <CSR3Matrix.h>
#include <iostream>
#include <cmath>
#include <utilities.h>

#include <MonteCarloSimulator.h>

MonteCarloSimulator::MonteCarloSimulator(MKL_INT basisSize,
MKL_INT samplesNumber, Solver &solver) :
		basisSize(basisSize), samplesNumber(samplesNumber), solver(solver), zeroVector(
				new COMPLEX_TYPE[basisSize]), groundState(
				new COMPLEX_TYPE[basisSize]) {
	for (int i = 0; i < basisSize; i++) {
		zeroVector[i].real = 0.0;
		zeroVector[i].imag = 0.0;
	}

	complex_cblas_copy(basisSize, zeroVector, NO_INC, groundState, NO_INC);
	groundState[0].real = 1.0;
}

MonteCarloSimulator::~MonteCarloSimulator() {
	delete[] zeroVector;
	delete[] groundState;
}

SimulationResult *MonteCarloSimulator::simulate(std::ostream &consoleStream) {
	//A storage of final states of all realizations
	COMPLEX_TYPE ** const result = new COMPLEX_TYPE*[samplesNumber];
	for (int i = 0; i < samplesNumber; i++) {
		result[i] = new COMPLEX_TYPE[basisSize];
	}

	//#pragma omp parallel for
	for (int sampleIndex = 0; sampleIndex < samplesNumber; sampleIndex++) {
		solver.solve(consoleStream, groundState, result[sampleIndex]);
	}

	SimulationResult *simulationResult = new SimulationResult(samplesNumber, basisSize, result);
	return simulationResult;
}

