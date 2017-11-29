/*
 * SimulationResult.h
 *
 *  Created on: Nov 29, 2017
 *      Author: fake_sci
 */

#ifndef SRC_INCLUDE_SIMULATIONRESULT_H_
#define SRC_INCLUDE_SIMULATIONRESULT_H_

#include <precision-definition.h>
#include <ImpreciseValue.h>

class SimulationResult {
	const int samplesNumber;
	const int basisSize;
	COMPLEX_TYPE ** const result;

	//caches
	mutable ImpreciseValue *photonNumber = nullptr;

public:
	SimulationResult(int samplesNumber, int basisSize,
			COMPLEX_TYPE ** const result);
	~SimulationResult();

	ImpreciseValue &getMeanPhotonNumber() const;
};



#endif /* SRC_INCLUDE_SIMULATIONRESULT_H_ */
