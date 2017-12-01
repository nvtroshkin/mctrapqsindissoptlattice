/*
 * RndNumProvider.cpp
 *
 *  Created on: Nov 30, 2017
 *      Author: fake_sci
 */

#include <precision-definition.h>
#include <mkl.h>
#include <RndNumProvider.h>

#include <RndNumProviderImpl.h>

RndNumProviderImpl::RndNumProviderImpl(int rndSeed, int nStreams) :
		nStreams(nStreams) {
	streams = new VSLStreamStatePtr[nStreams];
	for (int k = 0; k < nStreams; k++) {
		int status = vslNewStream(&streams[k], VSL_BRNG_MT2203 + k, rndSeed);
		if (status != VSL_STATUS_OK) {
			throw "Can't obtain a random numbers thread: " + status;
		}
	}
}

RndNumProviderImpl::~RndNumProviderImpl() {
	for (int i = 0; i < nStreams; i++) {
		vslDeleteStream(&streams[i]);
	}

	delete[] streams;
}

void RndNumProviderImpl::initBuffer(int streamId, FLOAT_TYPE *buffer,
		int bufferSize) {
	int status = vRngUniform(VSL_RNG_METHOD_UNIFORM_STD, streams[streamId],
			bufferSize, buffer, 0.0, 1.0);
	if (status != VSL_STATUS_OK) {
		throw "Can't fill a buffer with random numbers: " + status;
	}
}

