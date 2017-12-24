/*
 * MonteCarloSimulator.cpp
 *
 *  Created on: Nov 29, 2017
 *      Author: fakesci
 */

#include <omp.h>
#include <iostream>
#include <cmath>
#include <exception>
#include <vector>
#include <memory>

#include <cuda_runtime.h>
#include "helper_cuda.h"

#include "include/Model.h"
#include "include/MonteCarloSimulator.h"
#include "include/precision-definition.h"
#include "include/SimulationResult.h"
#include "Solver.h"
#include "utilities.h"
#include "SolverContext.h"

extern void simulate(const uint nBlocks, const uint nThreadsPerBlock,
		Solver * const * const solverDevPtrs);

MonteCarloSimulator::MonteCarloSimulator(uint samplesNumber, Model &model) :
		basisSize(model.getBasisSize()), samplesNumber(samplesNumber), model(
				model), groundState(new CUDA_COMPLEX_TYPE[basisSize]()) {
	groundState[0].x = 1.0;
}

MonteCarloSimulator::~MonteCarloSimulator() {
	delete[] groundState;
}

SimulationResult *MonteCarloSimulator::simulate(FLOAT_TYPE timeStep,
		uint nTimeSteps, uint threadsPerBlock, uint nBlocks) {

	SolverContext solverContext(nBlocks, timeStep, nTimeSteps, model);
	Solver ** solverDevPtrs = solverContext.createSolverDev(nBlocks,
			groundState);

	std::vector<CUDA_COMPLEX_TYPE *> * results = new std::vector<
	CUDA_COMPLEX_TYPE *>();
	results->reserve(samplesNumber);

	uint nIterations = (samplesNumber - 1) / nBlocks + 1;

	uint actualNBlocks;
	for (int n = 0; n < nIterations; ++n) {
		//to not calculate unnecessary samples at the end
		actualNBlocks = std::min(nBlocks, samplesNumber - n * nBlocks);

		//update initial state
		solverContext.initAllSolvers(groundState);

		::simulate(actualNBlocks, threadsPerBlock, solverDevPtrs);

		solverContext.appendAllResults(*results);
	}

	cudaFree(solverDevPtrs);
	//freeing resources
//	cudaDeviceReset();

//#if defined(DEBUG_CONTINUOUS) || defined(DEBUG_JUMPS)
//	print(std::cout, "Result", &(*results)[0], basisSize, samplesNumber);
//#endif

	SimulationResult *simulationResult = new SimulationResult(results,
			samplesNumber, model);

	return simulationResult;
}

