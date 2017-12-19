#include <eval-params.h>
#include <ImpreciseValue.h>
#include <Model.h>
#include <MonteCarloSimulator.h>
#include <system-constants.h>
#include <SimulationResult.h>
#include <iostream>
#include <chrono>

using namespace std;

int main(int argc, char **argv) {
	auto start = chrono::steady_clock::now();

	try {
		cout.precision(10);

		Model model(ATOM_1_LEVELS_NUMBER, ATOM_2_LEVELS_NUMBER,
				ATOM_3_LEVELS_NUMBER, FIELD_1_FOCK_STATES_NUMBER,
				FIELD_2_FOCK_STATES_NUMBER, FIELD_3_FOCK_STATES_NUMBER, KAPPA,
				DELTA_OMEGA, G, scE, J);
		MonteCarloSimulator monteCarloSimulator(MONTE_CARLO_SAMPLES_NUMBER,
				model);

		SimulationResult *result = monteCarloSimulator.simulate(TIME_STEP_SIZE,
				TIME_STEPS_NUMBER, THREADS_PER_BLOCK, N_BLOCKS);

		ImpreciseValue *firstCavityPhotons = result->getFirstCavityPhotons();
		cout << "Avg field photons in the first cavity: "
				<< firstCavityPhotons->mean << "; standard deviation: "
				<< firstCavityPhotons->standardDeviation << endl;

		ImpreciseValue *secondCavityPhotons = result->getSecondCavityPhotons();
		cout << "Avg field photons in the second cavity: "
				<< secondCavityPhotons->mean << "; standard deviation: "
				<< secondCavityPhotons->standardDeviation << endl;

		ImpreciseValue *thirdCavityPhotons = result->getThirdCavityPhotons();
		cout << "Avg field photons in the third cavity: "
				<< thirdCavityPhotons->mean << "; standard deviation: "
				<< thirdCavityPhotons->standardDeviation << endl;

		//freeing up resources
		delete result;
	} catch (const std::string &message) {
		cerr << message << endl;
	} catch (char *message) {
		cerr << message << endl;
	} catch (...) {
		cout << "Exception has been thrown - terminating" << endl;
	}

	auto end = chrono::steady_clock::now();
	auto diff = end - start;
	cout << "Elapsed time is :  " << chrono::duration_cast < chrono::nanoseconds
			> (diff).count() / 1000000000.0 << "s" << endl;

	return 0;
}
