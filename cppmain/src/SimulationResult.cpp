/*
 * SimulationResult.cpp
 *
 *  Created on: Nov 29, 2017
 *      Author: fakesci
 */

#include <cmath>
#include "include/precision-definition.h"
#include "include/Model.h"
#include "include/SimulationResult.h"
#include "mkl-constants.h"

SimulationResult::SimulationResult(const std::vector<CUDA_COMPLEX_TYPE * > * const results,
                                   const int samplesNumber, const Model &model) :
        results(*results), samplesNumber(samplesNumber), basisSize(
        model.getBasisSize()), model(model) {
}

SimulationResult::~SimulationResult() {
    delete avgPhotons1;
    delete avgPhotons2;
    delete avgPhotons3;

    for (int i = 0; i < samplesNumber; i++) {
        delete[] results[i];
    }
    delete &results;
}

ImpreciseValue *SimulationResult::getFirstCavityPhotons() const {
    if (avgPhotons1 == nullptr) {
        avgPhotons1 = getAvgPhotons(&Model::n1);
    }

    return avgPhotons1;
}

ImpreciseValue *SimulationResult::getSecondCavityPhotons() const {
    if (avgPhotons2 == nullptr) {
        avgPhotons2 = getAvgPhotons(&Model::n2);
    }

    return avgPhotons2;
}

ImpreciseValue *SimulationResult::getThirdCavityPhotons() const {
    if (avgPhotons3 == nullptr) {
        avgPhotons3 = getAvgPhotons(&Model::n3);
    }

    return avgPhotons3;
}

inline ImpreciseValue *SimulationResult::getAvgPhotons(
        PhotonNumberFuncP n) const {
    //an auxiliary array with photon numbers for each basis vector is needed
    COMPLEX_TYPE statePhotonNumbers[basisSize];
    for (int i = 0; i < basisSize; i++) {
        statePhotonNumbers[i] = {(model.*n)(i), 0.0};
    }

    //Sum(<psi|n|psi>)
    //mult psi on ns
    FLOAT_TYPE meanPhotonNumbers[samplesNumber];
    COMPLEX_TYPE norm2;
    COMPLEX_TYPE tempVector[basisSize];
    for (int i = 0; i < samplesNumber; i++) {
        complex_vMul(basisSize, results[i], statePhotonNumbers, tempVector);
        complex_cblas_dotc_sub(basisSize, tempVector, NO_INC, results[i], NO_INC,
                               &norm2);

        //store for the variance
        meanPhotonNumbers[i] = norm2.x;
    }

    FLOAT_TYPE meanPhotonsNumber;
    ippsSum_f(meanPhotonNumbers, samplesNumber, &meanPhotonsNumber);

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

        FLOAT_TYPE sum2;
        ippsSum_f(temp, samplesNumber, &sum2);

        //(a^2 + b^2 - 2 a b)
        FLOAT_TYPE sum = std::abs(
                sum1 + samplesNumber * meanPhotonsNumber * meanPhotonsNumber
                - sum2);

        standardDeviation = std::sqrt(
                sum / (samplesNumber * (samplesNumber - 1)));
    }

    return new ImpreciseValue{meanPhotonsNumber, standardDeviation};
}
