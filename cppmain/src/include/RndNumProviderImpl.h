/*
 * RndNumProvider.h
 *
 *  Created on: Nov 30, 2017
 *      Author: fakesci
 */

#ifndef SRC_INCLUDE_RNDNUMPROVIDERIMPL_H_
#define SRC_INCLUDE_RNDNUMPROVIDERIMPL_H_

#include <RndNumProvider.h>

class RndNumProviderImpl: public RndNumProvider {
	const int nStreams;

	VSLStreamStatePtr *streams;

public:
	RndNumProviderImpl(int rndSeed, int nStreams);
	~RndNumProviderImpl();

	void initBuffer(int streamId, FLOAT_TYPE *buffer, int bufferSize) override;
};



#endif /* SRC_INCLUDE_RNDNUMPROVIDERIMPL_H_ */
