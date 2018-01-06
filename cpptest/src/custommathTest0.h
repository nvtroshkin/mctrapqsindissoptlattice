/*
 * custommathTest0.h
 *
 *  Created on: Jan 5, 2018
 *      Author: fakesci
 */

#ifndef SRC_CUSTOMMATHTEST0_H_
#define SRC_CUSTOMMATHTEST0_H_

#include "precision-definition.h"

template<uint vSize, uint blockSize, uint ilpColumn>
void testMultVectorVector(const CUDA_COMPLEX_TYPE * const v1DevPtr,
		const CUDA_COMPLEX_TYPE * const v2DevPtr,
		CUDA_COMPLEX_TYPE * const resultDevPtr);

extern template void testMultVectorVector<8*1024, 64, 1>(const CUDA_COMPLEX_TYPE * const v1DevPtr,
		const CUDA_COMPLEX_TYPE * const v2DevPtr,
		CUDA_COMPLEX_TYPE * const resultDevPtr);

extern template void testMultVectorVector<8*1024, 64, 2>(const CUDA_COMPLEX_TYPE * const v1DevPtr,
		const CUDA_COMPLEX_TYPE * const v2DevPtr,
		CUDA_COMPLEX_TYPE * const resultDevPtr);

extern template void testMultVectorVector<8*1024, 96, 1>(const CUDA_COMPLEX_TYPE * const v1DevPtr,
		const CUDA_COMPLEX_TYPE * const v2DevPtr,
		CUDA_COMPLEX_TYPE * const resultDevPtr);

extern template void testMultVectorVector<8*1024, 64, 3>(const CUDA_COMPLEX_TYPE * const v1DevPtr,
		const CUDA_COMPLEX_TYPE * const v2DevPtr,
		CUDA_COMPLEX_TYPE * const resultDevPtr);

template<uint vSize, uint blockSize, uint ilpColumn, uint ilpRow>
void testMultMatrixVector(const CUDA_COMPLEX_TYPE * const matrixDevPtr,
		const CUDA_COMPLEX_TYPE * const vectorDevPtr,
		CUDA_COMPLEX_TYPE * const resultDevPtr);

extern template void testMultMatrixVector<1024, 64, 1, 1>(const CUDA_COMPLEX_TYPE * const v1DevPtr,
		const CUDA_COMPLEX_TYPE * const v2DevPtr,
		CUDA_COMPLEX_TYPE * const resultDevPtr);

extern template void testMultMatrixVector<8*1024, 256, 1, 1>(const CUDA_COMPLEX_TYPE * const v1DevPtr,
		const CUDA_COMPLEX_TYPE * const v2DevPtr,
		CUDA_COMPLEX_TYPE * const resultDevPtr);

extern template void testMultMatrixVector<8*1024, 256, 2, 1>(const CUDA_COMPLEX_TYPE * const v1DevPtr,
		const CUDA_COMPLEX_TYPE * const v2DevPtr,
		CUDA_COMPLEX_TYPE * const resultDevPtr);

extern template void testMultMatrixVector<8*1024, 384, 1, 1>(const CUDA_COMPLEX_TYPE * const v1DevPtr,
		const CUDA_COMPLEX_TYPE * const v2DevPtr,
		CUDA_COMPLEX_TYPE * const resultDevPtr);

extern template void testMultMatrixVector<8*1024, 256, 3, 1>(const CUDA_COMPLEX_TYPE * const v1DevPtr,
		const CUDA_COMPLEX_TYPE * const v2DevPtr,
		CUDA_COMPLEX_TYPE * const resultDevPtr);

extern template void testMultMatrixVector<8*1024, 256, 1, 2>(const CUDA_COMPLEX_TYPE * const v1DevPtr,
		const CUDA_COMPLEX_TYPE * const v2DevPtr,
		CUDA_COMPLEX_TYPE * const resultDevPtr);

extern template void testMultMatrixVector<8*1024, 256, 1, 3>(const CUDA_COMPLEX_TYPE * const v1DevPtr,
		const CUDA_COMPLEX_TYPE * const v2DevPtr,
		CUDA_COMPLEX_TYPE * const resultDevPtr);

extern template void testMultMatrixVector<8*1024, 128, 3, 3>(const CUDA_COMPLEX_TYPE * const v1DevPtr,
		const CUDA_COMPLEX_TYPE * const v2DevPtr,
		CUDA_COMPLEX_TYPE * const resultDevPtr);

template<uint vSize, uint blockSize, uint ilpColumn>
void testMultSparseMatrixVector(const CUDA_COMPLEX_TYPE * const valuesDevPtr,
		const int * const columnsDevPtr, const int * const rowIndexDevPtr,
		const CUDA_COMPLEX_TYPE * const vectorDevPtr,
		CUDA_COMPLEX_TYPE * const resultDevPtr);

extern template void testMultSparseMatrixVector<1024, 64, 1>(const CUDA_COMPLEX_TYPE * const valuesDevPtr,
		const int * const columnsDevPtr, const int * const rowIndexDevPtr,
		const CUDA_COMPLEX_TYPE * const vectorDevPtr,
		CUDA_COMPLEX_TYPE * const resultDevPtr);

extern template void testMultSparseMatrixVector<8*1024, 256, 1>(const CUDA_COMPLEX_TYPE * const valuesDevPtr,
		const int * const columnsDevPtr, const int * const rowIndexDevPtr,
		const CUDA_COMPLEX_TYPE * const vectorDevPtr,
		CUDA_COMPLEX_TYPE * const resultDevPtr);

extern template void testMultSparseMatrixVector<8*1024, 256, 2>(const CUDA_COMPLEX_TYPE * const valuesDevPtr,
		const int * const columnsDevPtr, const int * const rowIndexDevPtr,
		const CUDA_COMPLEX_TYPE * const vectorDevPtr,
		CUDA_COMPLEX_TYPE * const resultDevPtr);

extern template void testMultSparseMatrixVector<8*1024, 384, 1>(const CUDA_COMPLEX_TYPE * const valuesDevPtr,
		const int * const columnsDevPtr, const int * const rowIndexDevPtr,
		const CUDA_COMPLEX_TYPE * const vectorDevPtr,
		CUDA_COMPLEX_TYPE * const resultDevPtr);

extern template void testMultSparseMatrixVector<8*1024, 256, 3>(const CUDA_COMPLEX_TYPE * const valuesDevPtr,
		const int * const columnsDevPtr, const int * const rowIndexDevPtr,
		const CUDA_COMPLEX_TYPE * const vectorDevPtr,
		CUDA_COMPLEX_TYPE * const resultDevPtr);

#endif /* SRC_CUSTOMMATHTEST0_H_ */
