/*
 * SimulationResult.h
 *
 *  Created on: Nov 29, 2017
 *      Author: fakesci
 */

#ifndef SRC_INCLUDE_SIMULATIONRESULT_H_
#define SRC_INCLUDE_SIMULATIONRESULT_H_

#include "precision-definition.h"
#include "Model.h"
#include "ImpreciseValue.h"

class SimulationResult {

    typedef uint (Model::*PhotonNumberFuncP)(uint index) const;

    COMPLEX_TYPE **const result;
    const int samplesNumber;
    const int basisSize;

    Model &model;

    //caches
    mutable ImpreciseValue *avgPhotons1 = nullptr;
    mutable ImpreciseValue *avgPhotons2 = nullptr;
    mutable ImpreciseValue *avgPhotons3 = nullptr;

    ImpreciseValue *getAvgPhotons(PhotonNumberFuncP n) const;

public:
    SimulationResult(COMPLEX_TYPE **const result, int samplesNumber,
                     Model &model);

    ~SimulationResult();

    ImpreciseValue *getFirstCavityPhotons() const;

    ImpreciseValue *getSecondCavityPhotons() const;

    ImpreciseValue *getThirdCavityPhotons() const;
};

#endif /* SRC_INCLUDE_SIMULATIONRESULT_H_ */
