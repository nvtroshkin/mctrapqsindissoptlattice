/*
 * custommath.h
 *
 *  Created on: Jan 5, 2018
 *      Author: fakesci
 */

#ifndef SRC_INCLUDE_CUSTOMMATH_H_
#define SRC_INCLUDE_CUSTOMMATH_H_

#include "precision-definition.h"

template<uint vSize, uint blockSize, uint ilpColumn, bool conjugateV>
__device__ void multVectorVector(
		const CUDA_COMPLEX_TYPE * __restrict__ const v1,
		const CUDA_COMPLEX_TYPE * __restrict__ const v2,
		CUDA_COMPLEX_TYPE * __restrict__ const result);

template<uint vSize, uint blockSize, uint ilpColumn, uint ilpRow,
		bool conjugateV>
__device__ void multMatrixVector(
		const CUDA_COMPLEX_TYPE * __restrict__ const matrix,
		const CUDA_COMPLEX_TYPE * __restrict__ const vector,
		CUDA_COMPLEX_TYPE * __restrict__ const result);

template<uint vSize, uint blockSize, uint ilpColumn>
__device__ void multSparseMatrixVector(
		const CUDA_COMPLEX_TYPE * __restrict__ const values,
		const int * __restrict__ const columns,
		const int * __restrict__ const rowIndex,
		const CUDA_COMPLEX_TYPE * __restrict__ const vector,
		CUDA_COMPLEX_TYPE * __restrict__ const result);

#endif /* SRC_INCLUDE_CUSTOMMATH_H_ */
