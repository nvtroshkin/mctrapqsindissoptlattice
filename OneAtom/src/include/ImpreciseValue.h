/*
 * ImpreciseValue.h
 *
 *  Created on: Nov 29, 2017
 *      Author: fake_sci
 */

#ifndef SRC_INCLUDE_IMPRECISEVALUE_H_
#define SRC_INCLUDE_IMPRECISEVALUE_H_

#include <precision-definition.h>

struct ImpreciseValue {
	const FLOAT_TYPE mean;
	const FLOAT_TYPE standardDeviation;

	ImpreciseValue(FLOAT_TYPE mean, FLOAT_TYPE standardDeviation) :
			mean(mean), standardDeviation(standardDeviation) {
	}
};

#endif /* SRC_INCLUDE_IMPRECISEVALUE_H_ */
