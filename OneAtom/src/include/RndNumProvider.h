/*
 * RndNumProvider.h
 *
 *  Created on: Nov 30, 2017
 *      Author: fakesci
 */

#ifndef SRC_INCLUDE_RNDNUMPROVIDER_H_
#define SRC_INCLUDE_RNDNUMPROVIDER_H_

#include <precision-definition.h>

struct RndNumProvider {
public:
	virtual ~RndNumProvider() {
	}

	/**
	 * Fills a buffer with random numbers from specified random numbers stream
	 * (for usage in a multithreaded environment)
	 */
	virtual void initBuffer(int streamId, FLOAT_TYPE *buffer,
			int bufferSize) = 0;

};

#endif /* SRC_INCLUDE_RNDNUMPROVIDER_H_ */
