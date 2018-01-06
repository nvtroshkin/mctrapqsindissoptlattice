/*
 * custommathTest.cpp
 *
 *  Created on: Jan 5, 2018
 *      Author: fakesci
 */

#include <cuda_runtime.h>
#include "helper_cuda.h"

#include "precision-definition.h"
#include "custommath.h"

template<uint vSize, uint blockSize, uint ilpColumn, bool conjugateV>
__global__ void testMultVectorVectorKernel(
		const CUDA_COMPLEX_TYPE * const v1DevPtr,
		const CUDA_COMPLEX_TYPE * const v2DevPtr,
		CUDA_COMPLEX_TYPE * const resultDevPtr) {
	multVectorVector<vSize, blockSize, ilpColumn, conjugateV>(v1DevPtr,
			v2DevPtr, resultDevPtr);
}

template<uint vSize, uint blockSize, uint ilpColumn, bool conjugateV>
void testMultVectorVector(const CUDA_COMPLEX_TYPE * const v1DevPtr,
		const CUDA_COMPLEX_TYPE * const v2DevPtr,
		CUDA_COMPLEX_TYPE * const resultDevPtr) {
testMultVectorVectorKernel<vSize, blockSize, ilpColumn, conjugateV><<<1, blockSize>>>(v1DevPtr, v2DevPtr, resultDevPtr);

					getLastCudaError("testMultVectorVector");
}

template void testMultVectorVector<8 * 1024, 64, 1, true>(
		const CUDA_COMPLEX_TYPE * const v1DevPtr,
		const CUDA_COMPLEX_TYPE * const v2DevPtr,
		CUDA_COMPLEX_TYPE * const resultDevPtr);

template void testMultVectorVector<8 * 1024, 64, 2, true>(
		const CUDA_COMPLEX_TYPE * const v1DevPtr,
		const CUDA_COMPLEX_TYPE * const v2DevPtr,
		CUDA_COMPLEX_TYPE * const resultDevPtr);

template void testMultVectorVector<8 * 1024, 96, 1, true>(
		const CUDA_COMPLEX_TYPE * const v1DevPtr,
		const CUDA_COMPLEX_TYPE * const v2DevPtr,
		CUDA_COMPLEX_TYPE * const resultDevPtr);

template void testMultVectorVector<8 * 1024, 64, 3, true>(
		const CUDA_COMPLEX_TYPE * const v1DevPtr,
		const CUDA_COMPLEX_TYPE * const v2DevPtr,
		CUDA_COMPLEX_TYPE * const resultDevPtr);

template void testMultVectorVector<8 * 1024, 64, 3, false>(
		const CUDA_COMPLEX_TYPE * const v1DevPtr,
		const CUDA_COMPLEX_TYPE * const v2DevPtr,
		CUDA_COMPLEX_TYPE * const resultDevPtr);

template<uint vSize, uint blockSize, uint ilpColumn, uint ilpRow, bool conjugateV>
__global__ void testMultMatrixVectorKernel(
		const CUDA_COMPLEX_TYPE * const matrixDevPtr,
		const CUDA_COMPLEX_TYPE * const vectorDevPtr,
		CUDA_COMPLEX_TYPE * const resultDevPtr) {
	multMatrixVector<vSize, blockSize, ilpColumn, ilpRow, conjugateV>(matrixDevPtr,
			vectorDevPtr, resultDevPtr);
}

template<uint vSize, uint blockSize, uint ilpColumn, uint ilpRow, bool conjugateV>
void testMultMatrixVector(const CUDA_COMPLEX_TYPE * const matrixDevPtr,
		const CUDA_COMPLEX_TYPE * const vectorDevPtr,
		CUDA_COMPLEX_TYPE * const resultDevPtr) {
testMultMatrixVectorKernel<vSize, blockSize, ilpColumn, ilpRow, conjugateV><<<1, blockSize>>>(matrixDevPtr, vectorDevPtr, resultDevPtr);

					getLastCudaError("testMultVectorVector");
}

template void testMultMatrixVector<1024, 64, 1, 1, false>(
		const CUDA_COMPLEX_TYPE * const v1DevPtr,
		const CUDA_COMPLEX_TYPE * const v2DevPtr,
		CUDA_COMPLEX_TYPE * const resultDevPtr);

template void testMultMatrixVector<8 * 1024, 256, 1, 1, false>(
		const CUDA_COMPLEX_TYPE * const v1DevPtr,
		const CUDA_COMPLEX_TYPE * const v2DevPtr,
		CUDA_COMPLEX_TYPE * const resultDevPtr);

template void testMultMatrixVector<8 * 1024, 256, 2, 1, false>(
		const CUDA_COMPLEX_TYPE * const v1DevPtr,
		const CUDA_COMPLEX_TYPE * const v2DevPtr,
		CUDA_COMPLEX_TYPE * const resultDevPtr);

template void testMultMatrixVector<8 * 1024, 384, 1, 1, false>(
		const CUDA_COMPLEX_TYPE * const v1DevPtr,
		const CUDA_COMPLEX_TYPE * const v2DevPtr,
		CUDA_COMPLEX_TYPE * const resultDevPtr);

template void testMultMatrixVector<8 * 1024, 256, 3, 1, false>(
		const CUDA_COMPLEX_TYPE * const v1DevPtr,
		const CUDA_COMPLEX_TYPE * const v2DevPtr,
		CUDA_COMPLEX_TYPE * const resultDevPtr);

template void testMultMatrixVector<8 * 1024, 256, 1, 2, false>(
		const CUDA_COMPLEX_TYPE * const v1DevPtr,
		const CUDA_COMPLEX_TYPE * const v2DevPtr,
		CUDA_COMPLEX_TYPE * const resultDevPtr);

template void testMultMatrixVector<8 * 1024, 256, 1, 3, false>(
		const CUDA_COMPLEX_TYPE * const v1DevPtr,
		const CUDA_COMPLEX_TYPE * const v2DevPtr,
		CUDA_COMPLEX_TYPE * const resultDevPtr);

template void testMultMatrixVector<8 * 1024, 128, 3, 3, false>(
		const CUDA_COMPLEX_TYPE * const v1DevPtr,
		const CUDA_COMPLEX_TYPE * const v2DevPtr,
		CUDA_COMPLEX_TYPE * const resultDevPtr);

template void testMultMatrixVector<8 * 1024, 128, 3, 3, true>(
		const CUDA_COMPLEX_TYPE * const v1DevPtr,
		const CUDA_COMPLEX_TYPE * const v2DevPtr,
		CUDA_COMPLEX_TYPE * const resultDevPtr);

template<uint vSize, uint blockSize, uint ilpColumn>
__global__ void testMultSparseMatrixVectorKernel(
		const CUDA_COMPLEX_TYPE * const valuesDevPtr,
		const int * const columnsDevPtr, const int * const rowIndexDevPtr,
		const CUDA_COMPLEX_TYPE * const vectorDevPtr,
		CUDA_COMPLEX_TYPE * const resultDevPtr) {
	multSparseMatrixVector<vSize, blockSize, ilpColumn>(valuesDevPtr,
			columnsDevPtr, rowIndexDevPtr, vectorDevPtr, resultDevPtr);
}

template<uint vSize, uint blockSize, uint ilpColumn>
void testMultSparseMatrixVector(const CUDA_COMPLEX_TYPE * const valuesDevPtr,
		const int * const columnsDevPtr, const int * const rowIndexDevPtr,
		const CUDA_COMPLEX_TYPE * const vectorDevPtr,
		CUDA_COMPLEX_TYPE * const resultDevPtr) {
testMultSparseMatrixVectorKernel<vSize, blockSize, ilpColumn><<<1, blockSize>>>(valuesDevPtr, columnsDevPtr, rowIndexDevPtr, vectorDevPtr, resultDevPtr);

					getLastCudaError("testMultVectorVector");
}

template void testMultSparseMatrixVector<1024, 64, 1>(
		const CUDA_COMPLEX_TYPE * const valuesDevPtr,
		const int * const columnsDevPtr, const int * const rowIndexDevPtr,
		const CUDA_COMPLEX_TYPE * const vectorDevPtr,
		CUDA_COMPLEX_TYPE * const resultDevPtr);

template void testMultSparseMatrixVector<8 * 1024, 256, 1>(
		const CUDA_COMPLEX_TYPE * const valuesDevPtr,
		const int * const columnsDevPtr, const int * const rowIndexDevPtr,
		const CUDA_COMPLEX_TYPE * const vectorDevPtr,
		CUDA_COMPLEX_TYPE * const resultDevPtr);

template void testMultSparseMatrixVector<8 * 1024, 256, 2>(
		const CUDA_COMPLEX_TYPE * const valuesDevPtr,
		const int * const columnsDevPtr, const int * const rowIndexDevPtr,
		const CUDA_COMPLEX_TYPE * const vectorDevPtr,
		CUDA_COMPLEX_TYPE * const resultDevPtr);

template void testMultSparseMatrixVector<8 * 1024, 384, 1>(
		const CUDA_COMPLEX_TYPE * const valuesDevPtr,
		const int * const columnsDevPtr, const int * const rowIndexDevPtr,
		const CUDA_COMPLEX_TYPE * const vectorDevPtr,
		CUDA_COMPLEX_TYPE * const resultDevPtr);

template void testMultSparseMatrixVector<8 * 1024, 256, 3>(
		const CUDA_COMPLEX_TYPE * const valuesDevPtr,
		const int * const columnsDevPtr, const int * const rowIndexDevPtr,
		const CUDA_COMPLEX_TYPE * const vectorDevPtr,
		CUDA_COMPLEX_TYPE * const resultDevPtr);
