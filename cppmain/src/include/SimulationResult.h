/*
 * SimulationResult.h
 *
 *  Created on: Nov 29, 2017
 *      Author: fakesci
 */

#ifndef SRC_INCLUDE_SIMULATIONRESULT_H_
#define SRC_INCLUDE_SIMULATIONRESULT_H_

#include <vector>

#include "precision-definition.h"
#include "Model.h"
#include "ImpreciseValue.h"

class SimulationResult {

    typedef uint (Model::*PhotonNumberFuncP)(uint index) const;

    const std::vector<CUDA_COMPLEX_TYPE * > &results;
    const int samplesNumber;
    const int basisSize;

    const Model &model;

    //caches
    mutable ImpreciseValue *avgPhotons1 = nullptr;
    mutable ImpreciseValue *avgPhotons2 = nullptr;
    mutable ImpreciseValue *avgPhotons3 = nullptr;

    ImpreciseValue *getAvgPhotons(PhotonNumberFuncP n) const;

public:
    SimulationResult(const std::vector<CUDA_COMPLEX_TYPE * > * const results, const int samplesNumber,
                     const Model &model);

    ~SimulationResult();

    ImpreciseValue *getFirstCavityPhotons() const;

    ImpreciseValue *getSecondCavityPhotons() const;

    ImpreciseValue *getThirdCavityPhotons() const;
};

#endif /* SRC_INCLUDE_SIMULATIONRESULT_H_ */
