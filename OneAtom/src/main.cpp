#include <iostream>
#include <fstream>
#include <math.h>
#include <cmath>
#include <chrono>
#include <omp.h>

#include <eval-params.h>
#include <mkl-constants.h>
#include <ModelBuilder.h>
#include <precision-definition.h>
#include <system-constants.h>
#include <Solver.h>
#include <MonteCarloSimulator.h>
#include <Solver.h>
#include <ImpreciseValue.h>
#include <SimulationResult.h>
#include <RndNumProviderImpl.h>

using namespace std;

int main(int argc, char **argv) {
	auto start = chrono::steady_clock::now();

	ModelBuilder modelBuilder(MAX_PHOTON_NUMBER, DRESSED_BASIS_SIZE, KAPPA,
			DELTA_OMEGA, G, LATIN_E);
	RndNumProviderImpl rndNumProvider(RANDSEED, 1);
	Solver solver(DRESSED_BASIS_SIZE, TIME_STEP_SIZE, TIME_STEPS_NUMBER,
			modelBuilder, rndNumProvider);

	MonteCarloSimulator monteCarloSimulator(DRESSED_BASIS_SIZE,
			MONTE_CARLO_SAMPLES_NUMBER, solver);
	SimulationResult *result = monteCarloSimulator.simulate(cout);

	ImpreciseValue photonNumber = result->getMeanPhotonNumber();
	cout << "Mean photon number: " << photonNumber.mean << "\n";
	cout << "Standard deviation: " << photonNumber.standardDeviation << endl;

	//freeing up resources
	delete result;

	auto end = chrono::steady_clock::now();
	auto diff = end - start;
	cout << "Elapsed time is :  " << chrono::duration_cast < chrono::nanoseconds
			> (diff).count() / 1000000000.0 << "s" << endl;

	return 0;
}
