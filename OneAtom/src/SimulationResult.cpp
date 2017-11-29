/*
 * SimulationResult.cpp
 *
 *  Created on: Nov 29, 2017
 *      Author: fake_sci
 */

#include <precision-definition.h>
#include <ImpreciseValue.h>
#include <utilities.h>
#include <mkl-constants.h>
#include <cmath>

#include <SimulationResult.h>

SimulationResult::SimulationResult(int samplesNumber, int basisSize,
COMPLEX_TYPE ** const result) :
		samplesNumber(samplesNumber), basisSize(basisSize), result(result) {
}

SimulationResult::~SimulationResult() {
	delete photonNumber;
	delete result;
}

ImpreciseValue &SimulationResult::getMeanPhotonNumber() const {
	if (photonNumber == nullptr) {
		//an auxiliary array with photon numbers for each basis vector is needed
		COMPLEX_TYPE statePhotonNumbers[basisSize];
		for (int i = 0; i < basisSize; i++) {
			statePhotonNumbers[i] = {n(i),0.0};
		}

		//Sum(<psi|n|psi>)
		//mult psi on ns
		FLOAT_TYPE meanPhotonNumbers[samplesNumber];
		COMPLEX_TYPE norm2;
		COMPLEX_TYPE tempVector[basisSize];
		for (int i = 0; i < samplesNumber; i++) {
			complex_vMul(basisSize, result[i], statePhotonNumbers, tempVector);
			complex_cblas_dotc_sub(basisSize, tempVector, NO_INC, result[i],
					NO_INC, &norm2);

			//store for the variance
			meanPhotonNumbers[i] = norm2.real;
		}

		FLOAT_TYPE meanPhotonsNumber = cblas_asum(samplesNumber,
				meanPhotonNumbers, NO_INC);
		meanPhotonsNumber /= samplesNumber;

		FLOAT_TYPE standardDeviation = 0.0;
		if (samplesNumber > 1) {
			//variance. Calculate like this to avoid close numbers subtraction
			//Sum(mean photon numbers)^2
			FLOAT_TYPE sum1 = cblas_dot(samplesNumber, meanPhotonNumbers,
					NO_INC, meanPhotonNumbers, NO_INC);

			//Sum(2*mean photon number*mean photon number[i])
			FLOAT_TYPE temp[basisSize];
			cblas_copy(samplesNumber, meanPhotonNumbers, NO_INC, temp, NO_INC);
			cblas_scal(samplesNumber, 2.0 * meanPhotonsNumber, temp, NO_INC);
			FLOAT_TYPE sum2 = cblas_asum(samplesNumber, temp, NO_INC);

			//(a^2 + b^2 - 2 a b)
			FLOAT_TYPE sum = abs(
					sum1 + samplesNumber * meanPhotonsNumber * meanPhotonsNumber
							- sum2);

			standardDeviation = sqrtf(
					sum / (samplesNumber * (samplesNumber - 1)));
		}

		photonNumber =
				new ImpreciseValue { meanPhotonsNumber, standardDeviation };
	}

	return *photonNumber;
}
