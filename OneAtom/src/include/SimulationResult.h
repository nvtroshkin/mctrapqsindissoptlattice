/*
 * SimulationResult.h
 *
 *  Created on: Nov 29, 2017
 *      Author: fakesci
 */

#ifndef SRC_INCLUDE_SIMULATIONRESULT_H_
#define SRC_INCLUDE_SIMULATIONRESULT_H_

#include <precision-definition.h>
#include <ImpreciseValue.h>
#include <Model.h>

class SimulationResult {

	typedef int (Model::*PhotonNumberFuncP)(int index) const;

	COMPLEX_TYPE ** const result;
	const int samplesNumber;
	const int basisSize;

	Model &model;

	//caches
	mutable ImpreciseValue *avgPhotons1 = nullptr;
	mutable ImpreciseValue *avgPhotons2 = nullptr;

	ImpreciseValue *getAvgPhotons(PhotonNumberFuncP n) const;

public:
	SimulationResult(COMPLEX_TYPE ** const result, int samplesNumber,
			Model &model);
	~SimulationResult();

	ImpreciseValue *getAvgFirstCavityPhotons() const;
	ImpreciseValue *getAvgSecondCavityPhotons() const;
};

#endif /* SRC_INCLUDE_SIMULATIONRESULT_H_ */