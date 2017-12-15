/*
 * MonteCarloSimulatorTest.cpp
 *
 *  Created on: Dec 8, 2017
 *      Author: fakesci
 */

#include "definitions.h"

#include "Model.h"
#include "Solver.h"
#include "MonteCarloSimulator.h"
#include "RndNumProviderImpl.h"

/**
 *	KAPPA = 1.0
 *	DELTA_OMEGA = 20.0
 *	G = 50.0
 *	LATIN_E = 10.0
 *	J = 0.1
 *
 *	atom1SSize = atom2SSize = atom3SSize = 2
 *	field1SSize = field2SSize = field3SSize = 2
 *
 *	timeStep = 0.00001
 *	timeStepsNumber = 50000
 *
 *	samples = 100
 *	randSeed = 777
 *
 */
TEST (MonteCarloSimulator, test) {
	std::ostringstream output;
	output.precision(10);

	Model model(2, 2, 2, 2, 2, 2, 1.0, 20.0, 50.0, 30.0, 0.1);
	RndNumProviderImpl rndNumProvider(777, THREADS_NUM);
	MonteCarloSimulator monteCarloSimulator(100, THREADS_NUM, model,
			rndNumProvider);

	SimulationResult *result = monteCarloSimulator.simulate(output, 0.00001,
			50000);

	const ImpreciseValue *firstCavityPhotons = result->getFirstCavityPhotons();
	const ImpreciseValue *secondCavityPhotons =
			result->getSecondCavityPhotons();
	const ImpreciseValue *thirdCavityPhotons = result->getThirdCavityPhotons();

	output << "Avg photons in the FIRST cavity: " << firstCavityPhotons->mean
			<< "; Standard deviation: " << firstCavityPhotons->standardDeviation
			<< std::endl;
	output << "Avg photons in the SECOND cavity: " << secondCavityPhotons->mean
			<< "; Standard deviation: "
			<< secondCavityPhotons->standardDeviation << std::endl;
	output << "Avg photons in the THIRD cavity: " << thirdCavityPhotons->mean
			<< "; Standard deviation: " << thirdCavityPhotons->standardDeviation
			<< std::endl;

	std::cout << output.str();

	//in a jump occurs a significant loss of precision, because
	// 1) the method of detection of the time of a jump is rough (just using a previous step)
	// 2) after a jump a state vector gets 1/100 of its previous norm with 2 significant digits cut

	FLOAT_TYPE expectedMeanPhotonsFirst = 0.06314346804;
	FLOAT_TYPE expectedMeanPhotonsSecond = 0.06319560942;
	FLOAT_TYPE expectedMeanPhotonsThird = 0.06388715297;

	//The expected values should lie in the confidence interval
	ASSERT_TRUE(
			std::abs(firstCavityPhotons->mean - expectedMeanPhotonsFirst)
					< firstCavityPhotons->standardDeviation);
	ASSERT_TRUE(
			std::abs(secondCavityPhotons->mean - expectedMeanPhotonsSecond)
					< secondCavityPhotons->standardDeviation);
	ASSERT_TRUE(
			std::abs(thirdCavityPhotons->mean - expectedMeanPhotonsThird)
					< thirdCavityPhotons->standardDeviation);

	//tight?
	ASSERT_THAT(firstCavityPhotons->mean, FloatEq8digits(expectedMeanPhotonsFirst));
	ASSERT_THAT(firstCavityPhotons->standardDeviation,
			FloatEq8digits(0.002187166471));

	ASSERT_THAT(secondCavityPhotons->mean, FloatEq8digits(expectedMeanPhotonsSecond));
	ASSERT_THAT(secondCavityPhotons->standardDeviation,
			FloatEq8digits(0.002238600222));

	ASSERT_THAT(thirdCavityPhotons->mean, FloatEq8digits(expectedMeanPhotonsThird));
	ASSERT_THAT(thirdCavityPhotons->standardDeviation,
			FloatEq8digits(0.002930851397));

	delete result;
}

