#include <eval-params.h>
#include <ImpreciseValue.h>
#include <Model.h>
#include <MonteCarloSimulator.h>
#include <RndNumProviderImpl.h>
#include <system-constants.h>
#include <SimulationResult.h>
#include <iostream>
#include <chrono>

using namespace std;

int main(int argc, char **argv) {
	auto start = chrono::steady_clock::now();

	try {

		Model model(ATOM_1_LEVELS_NUMBER, ATOM_2_LEVELS_NUMBER,
				FIELD_1_FOCK_STATES_NUMBER, FIELD_2_FOCK_STATES_NUMBER, KAPPA,
				DELTA_OMEGA, G, scE, J);
		RndNumProviderImpl rndNumProvider(RANDSEED, THREADS_NUM);
		MonteCarloSimulator monteCarloSimulator(MONTE_CARLO_SAMPLES_NUMBER,
		THREADS_NUM, model, rndNumProvider);

		SimulationResult *result = monteCarloSimulator.simulate(cout,
				TIME_STEP_SIZE, TIME_STEPS_NUMBER);

		ImpreciseValue *firstCavityPhotons =
				result->getFirstCavityPhotons();
		cout << "Avg field photons in the first cavity: "
				<< firstCavityPhotons->mean << "; standard deviation: "
				<< firstCavityPhotons->standardDeviation << endl;

		ImpreciseValue *secondCavityPhotons =
				result->getSecondCavityPhotons();
		cout << "Avg field photons in the second cavity: "
				<< secondCavityPhotons->mean << "; standard deviation: "
				<< secondCavityPhotons->standardDeviation << endl;

		//freeing up resources
		delete result;
	} catch (char *message) {
		cerr << message;
	}

	auto end = chrono::steady_clock::now();
	auto diff = end - start;
	cout << "Elapsed time is :  " << chrono::duration_cast < chrono::nanoseconds
			> (diff).count() / 1000000000.0 << "s" << endl;

	return 0;
}
