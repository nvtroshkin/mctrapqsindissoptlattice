/*
 * MonteCarloSimulator.cpp
 *
 *  Created on: Nov 29, 2017
 *      Author: fakesci
 */

#include <mkl-constants.h>
#include <MonteCarloSimulator.h>
#include <omp.h>
#include <iostream>
#include <cmath>

#include <precision-definition.h>

MonteCarloSimulator::MonteCarloSimulator(MKL_INT basisSize,
MKL_INT samplesNumber, int nThreads, ModelBuilder &modelBuilder,
		RndNumProvider &rndNumProvider) :
		basisSize(basisSize), samplesNumber(samplesNumber), nThreads(nThreads), modelBuilder(
				modelBuilder), rndNumProvider(rndNumProvider), zeroVector(
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

SimulationResult *MonteCarloSimulator::simulate(std::ostream &consoleStream,
FLOAT_TYPE timeStep, int nTimeSteps) {
	//A storage of final states of all realizations
	COMPLEX_TYPE ** const result = new COMPLEX_TYPE*[samplesNumber];
	for (int i = 0; i < samplesNumber; i++) {
		result[i] = new COMPLEX_TYPE[basisSize];
	}

	//to obtain unbiased random numbers for each thread
	int threadId = 0;

#ifdef PRINT_PROGRESS
	int progress = 0;
#endif

#if THREADS_NUM>1
#pragma omp parallel num_threads(THREADS_NUM) private(threadId)
	{
		threadId = omp_get_thread_num();
#ifdef DEBUG
		consoleStream << "Thread " + std::to_string(threadId) + " started\n";
#endif
#endif
	Solver solver(threadId, basisSize, timeStep, nTimeSteps, modelBuilder,
			rndNumProvider);

#if THREADS_NUM>1
#pragma omp for
#endif
	for (int sampleIndex = 0; sampleIndex < samplesNumber; sampleIndex++) {
		solver.solve(consoleStream, groundState, result[sampleIndex]);
#ifdef PRINT_PROGRESS
		if(progress % NOTIFY_EACH_N_SAMPLES == 0) {
			consoleStream << "Progress: " + std::to_string(std::lround(100.0*progress/samplesNumber)) + "%\n";
		}
#pragma omp atomic update
		progress++;
#endif
	}

#if THREADS_NUM>1
#ifdef DEBUG
	consoleStream << "Thread: " + std::to_string(threadId) + " finished\n";
#endif
}
#endif
	SimulationResult *simulationResult = new SimulationResult(samplesNumber,
			basisSize, result);
	return simulationResult;
}

