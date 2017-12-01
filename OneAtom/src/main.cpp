#include <eval-params.h>
#include <ImpreciseValue.h>
#include <ModelBuilder.h>
#include <MonteCarloSimulator.h>
#include <RndNumProviderImpl.h>
#include <system-constants.h>
#include <SimulationResult.h>
#include <iostream>
#include <chrono>

using namespace std;

int main(int argc, char **argv) {
	auto start = chrono::steady_clock::now();

	ModelBuilder modelBuilder(MAX_PHOTON_NUMBER, DRESSED_BASIS_SIZE, KAPPA,
			DELTA_OMEGA, G, LATIN_E);
	RndNumProviderImpl rndNumProvider(RANDSEED, THREADS_NUM);
	MonteCarloSimulator monteCarloSimulator(DRESSED_BASIS_SIZE,
			MONTE_CARLO_SAMPLES_NUMBER, THREADS_NUM, modelBuilder, rndNumProvider);

	SimulationResult *result = monteCarloSimulator.simulate(cout, TIME_STEP_SIZE, TIME_STEPS_NUMBER);

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
