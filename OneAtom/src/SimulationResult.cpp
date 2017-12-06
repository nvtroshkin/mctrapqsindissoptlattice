/*
 * SimulationResult.cpp
 *
 *  Created on: Nov 29, 2017
 *      Author: fakesci
 */

#include <cmath>

#include <mkl-constants.h>
#include <SimulationResult.h>

SimulationResult::SimulationResult(COMPLEX_TYPE ** const result,
		int samplesNumber, Model &model) :
		result(result), samplesNumber(samplesNumber), basisSize(
				model.getBasisSize()), model(model) {
}

SimulationResult::~SimulationResult() {
	delete avgPhotons1;
	delete avgPhotons2;

	for (int i = 0; i < samplesNumber; i++) {
		delete[] result[i];
	}
	delete[] result;
}

ImpreciseValue *SimulationResult::getAvgFirstCavityPhotons() const {
	if (avgPhotons1 == nullptr) {
		avgPhotons1 = getAvgPhotons(&Model::n1);
	}

	return avgPhotons1;
}

ImpreciseValue *SimulationResult::getAvgSecondCavityPhotons() const {
	if (avgPhotons2 == nullptr) {
		avgPhotons2 = getAvgPhotons(&Model::n2);
	}

	return avgPhotons2;
}

inline ImpreciseValue *SimulationResult::getAvgPhotons(
		PhotonNumberFuncP n) const {
	//an auxiliary array with photon numbers for each basis vector is needed
	COMPLEX_TYPE statePhotonNumbers[basisSize];
	for (int i = 0; i < basisSize; i++) {
		statePhotonNumbers[i] = {(model.*n)(i),0.0};
	}

	//Sum(<psi|n|psi>)
	//mult psi on ns
	FLOAT_TYPE meanPhotonNumbers[samplesNumber];
	COMPLEX_TYPE norm2;
	COMPLEX_TYPE tempVector[basisSize];
	for (int i = 0; i < samplesNumber; i++) {
		complex_vMul(basisSize, result[i], statePhotonNumbers, tempVector);
		complex_cblas_dotc_sub(basisSize, tempVector, NO_INC, result[i], NO_INC,
				&norm2);

		//store for the variance
		meanPhotonNumbers[i] = norm2.real;
	}

	FLOAT_TYPE meanPhotonsNumber = cblas_asum(samplesNumber, meanPhotonNumbers,
			NO_INC);
	meanPhotonsNumber /= samplesNumber;

	FLOAT_TYPE standardDeviation = 0.0;
	if (samplesNumber > 1) {
		//variance. Calculate like this to avoid close numbers subtraction
		//Sum(mean photon numbers)^2
		FLOAT_TYPE sum1 = cblas_dot(samplesNumber, meanPhotonNumbers, NO_INC,
				meanPhotonNumbers, NO_INC);

		//Sum(2*mean photon number*mean photon number[i])
		FLOAT_TYPE temp[basisSize];
		cblas_copy(samplesNumber, meanPhotonNumbers, NO_INC, temp, NO_INC);
		cblas_scal(samplesNumber, 2.0 * meanPhotonsNumber, temp, NO_INC);
		FLOAT_TYPE sum2 = cblas_asum(samplesNumber, temp, NO_INC);

		//(a^2 + b^2 - 2 a b)
		FLOAT_TYPE sum = std::abs(
				sum1 + samplesNumber * meanPhotonsNumber * meanPhotonsNumber
						- sum2);

		standardDeviation = std::sqrt(
				sum / (samplesNumber * (samplesNumber - 1)));
	}

	return new ImpreciseValue { meanPhotonsNumber, standardDeviation };
}
