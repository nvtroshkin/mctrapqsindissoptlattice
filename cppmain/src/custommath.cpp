/*
 * custommath.cpp
 *
 *  Created on: Jan 5, 2018
 *      Author: fakesci
 */

#include "precision-definition.h"

template<typename T, typename V>
__device__ static inline void mulAdd(T &c1, const V &c2,
		const CUDA_COMPLEX_TYPE &c3) {
	c1.x += c2.x * c3.x - c2.y * c3.y;
	c1.y += c2.x * c3.y + c2.y * c3.x;
}

template<typename T, typename V>
__device__ static inline void add(T &c1, const V &c2) {
	c1.x += c2.x;
	c1.y += c2.y;
}

template<uint blockSize, uint ilpRow, uint reductionThreads>
__device__ inline void warpReduce(
		volatile CUDA_COMPLEX_TYPE * const __restrict__ sdata, const uint tid) {
	if (reductionThreads >= 64) {
#pragma unroll
		for (uint k = 0, s = tid; k < ilpRow; ++k, s += blockSize) {
			add(sdata[s], sdata[s + 32]);
		}
	}

	if (reductionThreads >= 32) {
#pragma unroll
		for (uint k = 0, s = tid; k < ilpRow; ++k, s += blockSize) {
			add(sdata[s], sdata[s + 16]);
		}
	}

	if (reductionThreads >= 16) {
#pragma unroll
		for (uint k = 0, s = tid; k < ilpRow; ++k, s += blockSize) {
			add(sdata[s], sdata[s + 8]);
		}
	}

	if (reductionThreads >= 8) {
#pragma unroll
		for (uint k = 0, s = tid; k < ilpRow; ++k, s += blockSize) {
			add(sdata[s], sdata[s + 4]);
		}
	}

	if (reductionThreads >= 4) {
#pragma unroll
		for (uint k = 0, s = tid; k < ilpRow; ++k, s += blockSize) {
			add(sdata[s], sdata[s + 2]);
		}
	}

	if (reductionThreads >= 2) {
#pragma unroll
		for (uint k = 0, s = tid; k < ilpRow; ++k, s += blockSize) {
			add(sdata[s], sdata[s + 1]);
		}
	}
}

template<uint vSize, uint blockSize, uint ilpRow, uint multIlpRowBlockSize,
		uint extraThreads, uint reductionThreads>
__device__ inline void gatherResults(const uint tid,
CUDA_COMPLEX_TYPE * const __restrict__ result,
CUDA_COMPLEX_TYPE * const __restrict__ sdata, const uint baseRowIndex) {

	__syncthreads();

//	if (/*tid >= 64 || tid < 32*/tid < 64) {
//#pragma unroll
//		for (uint ks = 0; ks < multIlpRowBlockSize; ks += blockSize) {
//			sdata[ks + tid].x = 0.0;
//			sdata[ks + tid].y = 0.0;
//		}
//	}
//	__syncthreads();

//Gather extra threads results to the power of 2 group
	if (tid < extraThreads) {
#pragma unroll
		for (uint k = 0, s = tid; k < ilpRow; ++k, s += blockSize) {
			add(sdata[s], sdata[s + reductionThreads]);
		}
	}

	__syncthreads();

	if (reductionThreads >= 512) {
		if (tid < 256) {
#pragma unroll
			for (uint k = 0, s = tid; k < ilpRow; ++k, s += blockSize) {
				add(sdata[s], sdata[s + 256]);
			}
		}

		__syncthreads();
	}

	if (reductionThreads >= 256) {
		if (tid < 128) {
#pragma unroll
			for (uint k = 0, s = tid; k < ilpRow; ++k, s += blockSize) {
				add(sdata[s], sdata[s + 128]);
			}
		}

		__syncthreads();
	}

	if (reductionThreads >= 128) {
		if (tid < 64) {
#pragma unroll
			for (uint k = 0, s = tid; k < ilpRow; ++k, s += blockSize) {
				add(sdata[s], sdata[s + 64]);
			}
		}

		__syncthreads();
	}

	if (tid < 32) {
		warpReduce<blockSize, ilpRow, reductionThreads>(sdata, tid);
	}

	if (tid == 0) {
#pragma unroll
		for (uint kr = 0, s = 0; kr < ilpRow; ++kr, s += blockSize) {
			result[baseRowIndex + kr] = sdata[s];
		}
	}

	__syncthreads();
}

template<uint startIndex, uint vSize, uint matrixSize, uint blockSize,
		uint ilpColumn, uint multIlpColumnIlpRow, uint multIlpColumnBlockSize>
__device__ void ilpByColumnRemainings(const uint tid,
		const CUDA_COMPLEX_TYPE * const __restrict__ matrix,
		const CUDA_COMPLEX_TYPE * const __restrict__ vector,
		CUDA_COMPLEX_TYPE * const __restrict__ sdata, const uint kmRow) {
//Here there are divergent endings of each row (not every thread has data to process)

	for (uint i = startIndex + tid; i < vSize; i += multIlpColumnBlockSize) {

		CUDA_COMPLEX_TYPE ilpRows[multIlpColumnIlpRow];

#pragma unroll
		for (int k = 0; k < multIlpColumnIlpRow; ++k) {
			ilpRows[k] = {0.0,0.0};
		}

		CUDA_COMPLEX_TYPE ilpV[ilpColumn];

#pragma unroll
		for (int k = 0; k < ilpColumn; ++k) {
			ilpV[k] = {0.0,0.0};
		}

#pragma unroll
		for (uint kc = 0, s = i; kc < ilpColumn; ++kc, s += blockSize) {
			if (s < vSize) {
#pragma unroll
				for (uint kr = 0, kmr = kmRow + s; kr < multIlpColumnIlpRow;
						kr += ilpColumn, kmr += vSize) {
					ilpRows[kr + kc] = matrix[kmr];
				}

				ilpV[kc] = vector[s];
			}
		}

#pragma unroll
		for (uint kr = 0, ks = tid; kr < multIlpColumnIlpRow;
				kr += ilpColumn, ks += blockSize) {
			FLOAT_TYPE f1 = 0.0, f2 = 0.0;

#pragma unroll
			for (uint kc = 0; kc < ilpColumn; ++kc) {
				CUDA_COMPLEX_TYPE rowValue = ilpRows[kr + kc];
				CUDA_COMPLEX_TYPE vectorValue = ilpV[kc];

				f1 += rowValue.x * vectorValue.x;
				f1 -= rowValue.y * vectorValue.y;
				f2 += rowValue.x * vectorValue.y;
				f2 += rowValue.y * vectorValue.x;
			}

			sdata[ks].x += f1;
			sdata[ks].y += f2;
		}
	}
}

template<uint endColumnIndex, uint vSize, uint blockSize, uint ilpColumn,
		uint multIlpColumnIlpRow, uint multIlpColumnBlockSize>
__device__ inline void ilpByColumn(const uint tid,
		const CUDA_COMPLEX_TYPE * const __restrict__ matrix,
		const CUDA_COMPLEX_TYPE * const __restrict__ vector,
		CUDA_COMPLEX_TYPE * const __restrict__ sdata, const uint kmRow) {

	for (uint i = tid; i < endColumnIndex; i += multIlpColumnBlockSize) {

		CUDA_COMPLEX_TYPE ilpRows[multIlpColumnIlpRow];
		CUDA_COMPLEX_TYPE ilpV[ilpColumn];

#pragma unroll
		for (uint kc = 0, s = i; kc < ilpColumn; ++kc, s += blockSize) {
#pragma unroll
			for (uint kr = 0, kmr = kmRow + s; kr < multIlpColumnIlpRow; kr +=
					ilpColumn, kmr += vSize) {
				ilpRows[kr + kc] = matrix[kmr];
//				if (s < vSize) {
//					ilpRows[kr + kc] = matrix[kmRow + s];
//				} else {
//					ilpRows[kr + kc] = {0.0,0.0};
//				}
			}

			ilpV[kc] = vector[s];
		}

//#pragma unroll
//		for (uint kc = 0, s = i; kc < ilpColumn; ++kc, s += blockSize) {
//			ilpV[kc] = vector[s];
//			if (s < vSize) {
//				ilpV[kc] = vector[s];
//			} else {
//				ilpV[kc] = {0.0,0.0};
//			}
//		}

#pragma unroll
		for (uint kr = 0, ks = 0; kr < multIlpColumnIlpRow;
				kr += ilpColumn, ks += blockSize) {
			FLOAT_TYPE f1 = 0.0, f2 = 0.0;

#pragma unroll
			for (uint kc = 0; kc < ilpColumn; ++kc) {
				CUDA_COMPLEX_TYPE rowValue = ilpRows[kr + kc];
				CUDA_COMPLEX_TYPE vectorValue = ilpV[kc];

				f1 += rowValue.x * vectorValue.x;
				f1 -= rowValue.y * vectorValue.y;
				f2 += rowValue.x * vectorValue.y;
				f2 += rowValue.y * vectorValue.x;
			}

			sdata[ks + tid].x += f1;
			sdata[ks + tid].y += f2;
		}
	}
}

template<uint startRowIndex, uint lastKMRow, uint remainColumns,
		uint endColumnIndex, uint vSize, uint blockSize, uint ilpColumn,
		uint ilpRow, uint multIlpColumnIlpRow, uint multIlpColumnBlockSize,
		uint multIlpRowBlockSize, uint extraThreads, uint reductionThreads>
__device__ inline void ilpByRowRemaining(const uint tid,
		const CUDA_COMPLEX_TYPE * const __restrict__ matrix,
		const CUDA_COMPLEX_TYPE * const __restrict__ vector,
		CUDA_COMPLEX_TYPE * const __restrict__ result,
		CUDA_COMPLEX_TYPE * const __restrict__ sdata) {
#pragma unroll
	for (uint ks = 0; ks < multIlpRowBlockSize; ks += blockSize) {
		sdata[ks + tid] = {0.0,0.0};
	}

	//the code could be as much as 10x slower if ILP by column doesn't fit the row size (perhaps, only if the ilp size is not a power of 2)
	ilpByColumnRemainings<0, vSize, vSize * vSize, blockSize, ilpColumn,
	multIlpColumnIlpRow, multIlpColumnBlockSize>(tid, matrix, vector,
			sdata, lastKMRow);

	gatherResults<vSize, blockSize, vSize - startRowIndex,
	multIlpRowBlockSize, extraThreads, reductionThreads>(tid, result, sdata, startRowIndex);
}

template<uint remainColumns, uint endColumnIndex, uint vSize, uint blockSize,
		uint ilpColumn, uint ilpRow, uint multIlpColumnIlpRow,
		uint multIlpColumnBlockSize, uint multIlpRowBlockSize,
		uint extraThreads, uint reductionThreads>
__device__ inline void multRowVector(const uint tid,
		const CUDA_COMPLEX_TYPE * const __restrict__ matrix,
		const CUDA_COMPLEX_TYPE * const __restrict__ vector,
		CUDA_COMPLEX_TYPE * const __restrict__ result,
		CUDA_COMPLEX_TYPE * const __restrict__ sdata, const uint kmRow,
		const uint row) {
#pragma unroll
	for (uint ks = 0; ks < multIlpRowBlockSize; ks += blockSize) {
		sdata[ks + tid] = {0.0,0.0};
	}

	//the code could be as much as 10x slower if ILP by column doesn't fit the row size (perhaps, only if the ilp size is not a power of 2)
	if(remainColumns) {
		ilpByColumn<endColumnIndex,vSize, blockSize, ilpColumn, multIlpColumnIlpRow,
		multIlpColumnBlockSize>(tid, matrix, vector, sdata, kmRow);

		ilpByColumnRemainings<endColumnIndex, vSize, vSize*vSize, blockSize,
		ilpColumn, multIlpColumnIlpRow, multIlpColumnBlockSize>(tid, matrix, vector, sdata, kmRow);
	} else {
		ilpByColumn<vSize,vSize, blockSize, ilpColumn, multIlpColumnIlpRow,
		multIlpColumnBlockSize>(tid, matrix, vector, sdata, kmRow);
	}

	gatherResults<vSize, blockSize, ilpRow,
	multIlpRowBlockSize, extraThreads, reductionThreads>(tid, result, sdata, row);
}

template<uint endRowIndex, uint vSize, uint blockSize, uint ilpColumn,
		uint ilpRow, uint multIlpColumnIlpRow, uint multIlpColumnBlockSize,
		uint multIlpRowBlockSize, uint remainColumns, uint endColumnIndex,
		uint kmRowInc, uint extraThreads, uint reductionThreads>
__device__ inline void ilpByRowBase(const uint tid,
		const CUDA_COMPLEX_TYPE * const __restrict__ matrix,
		const CUDA_COMPLEX_TYPE * const __restrict__ vector,
		CUDA_COMPLEX_TYPE * const __restrict__ result,
		CUDA_COMPLEX_TYPE * const __restrict__ sdata) {

	for (uint row = 0, kmRow = 0; row < endRowIndex; row += ilpRow, kmRow +=
			kmRowInc) {
		multRowVector<remainColumns, endColumnIndex, vSize, blockSize,
				ilpColumn, ilpRow, multIlpColumnIlpRow, multIlpColumnBlockSize,
				multIlpRowBlockSize, extraThreads, reductionThreads>(tid,
				matrix, vector, result, sdata, kmRow, row);
	}
}

__device__ constexpr uint _getNearestPowerOf2(const uint num, const uint temp) {
	return num < temp ? temp >> 1 : _getNearestPowerOf2(num, temp << 1);
}

__device__ constexpr uint getNearestPowerOf2(const uint num) {
	return num < 2 ? 0 : _getNearestPowerOf2(num, 1);
}

__device__ constexpr uint getReductionThreads(const uint num) {
	return getNearestPowerOf2(num);
}

template<uint vSize, uint blockSize, uint ilpColumn>
__device__ void multVectorVector(const CUDA_COMPLEX_TYPE * __restrict__ const v1,
		const CUDA_COMPLEX_TYPE * __restrict__ const v2,
		CUDA_COMPLEX_TYPE * __restrict__ const result) {

	constexpr uint multIlpColumnIlpRow = ilpColumn;
	constexpr uint multIlpRowBlockSize = blockSize;
	constexpr uint multIlpColumnBlockSize = ilpColumn * blockSize;

	constexpr uint remainColumns = vSize % multIlpColumnBlockSize;
	constexpr uint endColumnIndex = vSize - remainColumns;

	constexpr uint reductionThreads = getReductionThreads(blockSize);
	constexpr uint extraThreads = blockSize - reductionThreads;

	static __shared__ CUDA_COMPLEX_TYPE sdata[multIlpRowBlockSize];

	uint tid = threadIdx.x;

//the code could be as much as 10x slower if ILP by column doesn't fit the row size (perhaps,
//only if the ILP size is not a power of 2)
//both cases (extra rows and extra columns) are processed the same way (just fill gaps with zeros),
//it could be optimized though I don't know is it worth it

	multRowVector<remainColumns, endColumnIndex, vSize, blockSize, ilpColumn, 1,
			multIlpColumnIlpRow, multIlpColumnBlockSize, multIlpRowBlockSize,
			extraThreads, reductionThreads>(tid, v1, v2, result, sdata, 0, 0);
}

template<uint vSize, uint blockSize, uint ilpColumn, uint ilpRow>
__device__ void multMatrixVector(
		const CUDA_COMPLEX_TYPE * __restrict__ const matrix,
		const CUDA_COMPLEX_TYPE * __restrict__ const vector,
		CUDA_COMPLEX_TYPE * __restrict__ const result) {

	constexpr uint multIlpColumnIlpRow = ilpColumn * ilpRow;
	constexpr uint multIlpRowBlockSize = ilpRow * blockSize;
	constexpr uint multIlpColumnBlockSize = ilpColumn * blockSize;

	constexpr uint remainColumns = vSize % multIlpColumnBlockSize;
	constexpr uint remainRows = vSize % ilpRow;

	constexpr uint kmRowInc = vSize * ilpRow;

	constexpr uint endColumnIndex = vSize - remainColumns;
	constexpr uint endRowIndex = vSize - remainRows;

	constexpr uint matrixSize = vSize * vSize;

	constexpr uint lastKMRow = matrixSize - (matrixSize % kmRowInc);

	constexpr uint reductionThreads = getReductionThreads(blockSize);
	constexpr uint extraThreads = blockSize - reductionThreads;

	static __shared__ CUDA_COMPLEX_TYPE sdata[multIlpRowBlockSize];

	uint tid = threadIdx.x;

//the code could be as much as 10x slower if ILP by column doesn't fit the row size (perhaps,
//only if the ILP size is not a power of 2)
//both cases (extra rows and extra columns) are processed the same way (just fill gaps with zeros),
//it could be optimized though I don't know is it worth it
	if (remainRows) {
		ilpByRowBase<endRowIndex, vSize, blockSize, ilpColumn, ilpRow,
				multIlpColumnIlpRow, multIlpColumnBlockSize,
				multIlpRowBlockSize, remainColumns, endColumnIndex, kmRowInc,
				extraThreads, reductionThreads>(tid, matrix, vector, result,
				sdata);

		ilpByRowRemaining<endRowIndex, lastKMRow, remainColumns, endColumnIndex,
				vSize, blockSize, ilpColumn, ilpRow, multIlpColumnIlpRow,
				multIlpColumnBlockSize, multIlpRowBlockSize, extraThreads,
				reductionThreads>(tid, matrix, vector, result, sdata);
	} else {
		ilpByRowBase<vSize, vSize, blockSize, ilpColumn, ilpRow,
				multIlpColumnIlpRow, multIlpColumnBlockSize,
				multIlpRowBlockSize, remainColumns, endColumnIndex, kmRowInc,
				extraThreads, reductionThreads>(tid, matrix, vector, result,
				sdata);
	}
}

template<uint blockSize, uint ilpColumn, uint multIlpColumnIlpRow,
		uint multIlpColumnBlockSize>
__device__ void ilpByColumnRemainings(const uint tid, const uint startIndex,
		const uint rowSize, const CUDA_COMPLEX_TYPE * __restrict__ const values,
		const int * __restrict__ const columns,
		const CUDA_COMPLEX_TYPE * const __restrict__ vector,
		CUDA_COMPLEX_TYPE * const __restrict__ sdata, const uint kmRow) {

	//Here there are divergent endings of each row (not every thread has data to process)
	for (uint i = startIndex + tid; i < rowSize; i += multIlpColumnBlockSize) {

		int ilpColumns[ilpColumn] { };

#pragma unroll
		for (uint kc = 0, s = i; kc < ilpColumn; ++kc, s += blockSize) {
			if (s < rowSize) {
				ilpColumns[kc] = columns[kmRow + s];
			}
		}

		CUDA_COMPLEX_TYPE ilpRows[multIlpColumnIlpRow];

#pragma unroll
		for (int k = 0; k < multIlpColumnIlpRow; ++k) {
			ilpRows[k] = {0.0,0.0};
		}

		CUDA_COMPLEX_TYPE ilpV[ilpColumn];

#pragma unroll
		for (int k = 0; k < ilpColumn; ++k) {
			ilpV[k] = {0.0,0.0};
		}

#pragma unroll
		for (uint kc = 0, s = i; kc < ilpColumn; ++kc, s += blockSize) {
			if (s < rowSize) {
				ilpRows[kc] = values[kmRow + s];
				ilpV[kc] = vector[ilpColumns[kc]];
			}
		}

		FLOAT_TYPE f1 = 0.0, f2 = 0.0;

#pragma unroll
		for (uint kc = 0; kc < ilpColumn; ++kc) {
			f1 += ilpRows[kc].x * ilpV[kc].x;
			f1 -= ilpRows[kc].y * ilpV[kc].y;
			f2 += ilpRows[kc].x * ilpV[kc].y;
			f2 += ilpRows[kc].y * ilpV[kc].x;
		}

		sdata[tid].x += f1;
		sdata[tid].y += f2;
	}
}

template<uint blockSize, uint ilpColumn, uint multIlpColumnBlockSize>
__device__ inline void ilpByColumn(const uint tid, const uint rowSize,
		const CUDA_COMPLEX_TYPE * __restrict__ const values,
		const int * __restrict__ const columns,
		const CUDA_COMPLEX_TYPE * const __restrict__ vector,
		CUDA_COMPLEX_TYPE * const __restrict__ sdata, const uint kmRow) {

	for (uint i = tid; i < rowSize; i += multIlpColumnBlockSize) {

		int ilpColumns[ilpColumn];

#pragma unroll
		for (uint kc = 0, s = kmRow + i; kc < ilpColumn; ++kc, s += blockSize) {
			ilpColumns[kc] = columns[s];
		}

		CUDA_COMPLEX_TYPE ilpRows[ilpColumn];
		CUDA_COMPLEX_TYPE ilpV[ilpColumn];

#pragma unroll
		for (uint kc = 0, s = kmRow + i; kc < ilpColumn; ++kc, s += blockSize) {
			ilpRows[kc] = values[s];
//				if (s < vSize) {
//					ilpRows[kr + kc] = matrix[kmRow + s];
//				} else {
//					ilpRows[kr + kc] = {0.0,0.0};
//				}

			ilpV[kc] = vector[ilpColumns[kc]];
//			if (s < vSize) {
//				ilpV[kc] = vector[s];
//			} else {
//				ilpV[kc] = {0.0,0.0};
//			}
		}

		FLOAT_TYPE f1 = 0.0, f2 = 0.0;

#pragma unroll
		for (uint kc = 0; kc < ilpColumn; ++kc) {
			f1 += ilpRows[kc].x * ilpV[kc].x;
			f1 -= ilpRows[kc].y * ilpV[kc].y;
			f2 += ilpRows[kc].x * ilpV[kc].y;
			f2 += ilpRows[kc].y * ilpV[kc].x;
		}

		sdata[tid].x += f1;
		sdata[tid].y += f2;
	}
}

template<uint startRowIndex, uint vSize, uint blockSize, uint ilpColumn,
		uint multIlpColumnIlpRow, uint multIlpColumnBlockSize,
		uint multIlpRowBlockSize, uint extraThreads, uint reductionThreads>
__device__ inline void ilpByRowRemaining(const uint tid, const uint rowSize,
		const CUDA_COMPLEX_TYPE * __restrict__ const values,
		const int * __restrict__ const columns,
		const CUDA_COMPLEX_TYPE * const __restrict__ vector,
		CUDA_COMPLEX_TYPE * const __restrict__ result,
		CUDA_COMPLEX_TYPE * const __restrict__ sdata, const uint lastKMRow) {

#pragma unroll
	for (uint ks = 0; ks < multIlpRowBlockSize; ks += blockSize) {
		sdata[ks + tid] = {0.0,0.0};
	}

//the code could be as much as 10x slower if ILP by column doesn't fit the row size (perhaps, only if the ilp size is not a power of 2)
	ilpByColumnRemainings<blockSize, ilpColumn,
	multIlpColumnIlpRow, multIlpColumnBlockSize>(tid, 0, rowSize, values, columns, vector,
			sdata, lastKMRow);

	gatherResults<vSize, blockSize, vSize - startRowIndex,
	multIlpRowBlockSize, extraThreads, reductionThreads>(tid, result, sdata, startRowIndex);
}

template<uint vSize, uint blockSize, uint ilpColumn, uint ilpRow,
		uint multIlpColumnIlpRow, uint multIlpColumnBlockSize,
		uint multIlpRowBlockSize, uint extraThreads, uint reductionThreads>
__device__ inline void multRowVector(const uint tid,
		const CUDA_COMPLEX_TYPE * __restrict__ const values,
		const int * __restrict__ const columns,
		const int * __restrict__ const rowIndex,
		const CUDA_COMPLEX_TYPE * const __restrict__ vector,
		CUDA_COMPLEX_TYPE * const __restrict__ result,
		CUDA_COMPLEX_TYPE * const __restrict__ sdata, const uint row) {

//	constexpr uint ilpRowPlus1 = ilpRow + 1;
//
//	__shared__ const int ilpRowIndex[ilpRowPlus1];
//
//	if (tid < ilpRowPlus1) {
//		ilpRowIndex[tid] = rowIndex[tid];
//	}
//
//	__syncthreads();
//
//	//Assuming that all rows have the same size
//	int rowSize = 0;
//
//	if (tid == 0) {
//		for (int i = 0; i < ilpRow; ++i) {
//			int tempRowSize = ilpRowIndex[i + 1] - ilpRowIndex[i];
//			if (rowSize - tempRowSize < 0) {
//				rowSize = tempRowSize;
//			}
//			//assert/check on errors
//		}
//	}

//rowIndex is ready

	const int row_begin = rowIndex[row];
	const int rowSize = rowIndex[row + 1] - row_begin;

	const uint remainColumns = rowSize % multIlpColumnBlockSize;

#pragma unroll
	for (uint ks = 0; ks < multIlpRowBlockSize; ks += blockSize) {
		sdata[ks + tid] = {0.0,0.0};
	}

	//the code could be as much as 10x slower if ILP by column doesn't fit the row size (perhaps, only if the ilp size is not a power of 2)
	if (remainColumns) {
		const uint endColumnIndex = rowSize - remainColumns;

		ilpByColumn<blockSize, ilpColumn, multIlpColumnBlockSize>(tid,
				endColumnIndex, values, columns, vector, sdata, row_begin);

		ilpByColumnRemainings<blockSize, ilpColumn, multIlpColumnIlpRow,
				multIlpColumnBlockSize>(tid, endColumnIndex, rowSize, values,
				columns, vector, sdata, row_begin);
	} else {
		ilpByColumn<blockSize, ilpColumn, multIlpColumnBlockSize>(tid, rowSize,
				values, columns, vector, sdata, row_begin);
	}

	gatherResults<vSize, blockSize, ilpRow, multIlpRowBlockSize, extraThreads,
			reductionThreads>(tid, result, sdata, row);
}

template<uint endRowIndex, uint vSize, uint blockSize, uint ilpColumn,
		uint ilpRow, uint multIlpColumnIlpRow, uint multIlpColumnBlockSize,
		uint multIlpRowBlockSize, uint extraThreads, uint reductionThreads>
__device__ inline void ilpByRowBase(const uint tid,
		const CUDA_COMPLEX_TYPE * __restrict__ const values,
		const int * __restrict__ const columns,
		const int * __restrict__ const rowIndex,
		const CUDA_COMPLEX_TYPE * const __restrict__ vector,
		CUDA_COMPLEX_TYPE * const __restrict__ result,
		CUDA_COMPLEX_TYPE * const __restrict__ sdata) {

	for (uint row = 0; row < endRowIndex; ++row) {
		multRowVector<vSize, blockSize, ilpColumn, ilpRow, multIlpColumnIlpRow,
				multIlpColumnBlockSize, multIlpRowBlockSize, extraThreads,
				reductionThreads>(tid, values, columns, rowIndex, vector,
				result, sdata, row);
	}
}

template<uint vSize, uint blockSize, uint ilpColumn>
__device__ void multSparseMatrixVector(
		const CUDA_COMPLEX_TYPE * __restrict__ const values,
		const int * __restrict__ const columns,
		const int * __restrict__ const rowIndex,
		const CUDA_COMPLEX_TYPE * __restrict__ const vector,
		CUDA_COMPLEX_TYPE * __restrict__ const result) {

	constexpr uint ilpRow = 1;

	constexpr uint multIlpColumnIlpRow = ilpColumn * ilpRow;
	constexpr uint multIlpRowBlockSize = ilpRow * blockSize;
	constexpr uint multIlpColumnBlockSize = ilpColumn * blockSize;

//	constexpr uint remainRows = vSize % ilpRow;
//	constexpr uint endRowIndex = vSize - remainRows;

//	const uint lastKMRow = rowIndex[vSize - ilpRow];

	constexpr uint reductionThreads = getReductionThreads(blockSize);
	constexpr uint extraThreads = blockSize - reductionThreads;

	static __shared__ CUDA_COMPLEX_TYPE sdata[multIlpRowBlockSize];

	uint tid = threadIdx.x;

//the code could be as much as 10x slower if ILP by column doesn't fit the row size (perhaps,
//only if the ILP size is not a power of 2)
//both cases (extra rows and extra columns) are processed the same way (just fill gaps with zeros),
//it could be optimized though I don't know is it worth it

//ilpRow is not implemented
//	if (remainRows) {
//		ilpByRowBase<endRowIndex, vSize, blockSize, ilpColumn, ilpRow,
//				multIlpColumnIlpRow, multIlpColumnBlockSize,
//				multIlpRowBlockSize, extraThreads, reductionThreads>(tid,
//				values, columns, rowIndex, vector, result, ilpRows, ilpV,
//				sdata);

//No ILP by row

//		ilpByRowRemaining<endRowIndex, vSize, blockSize, ilpColumn,
//				multIlpColumnIlpRow, multIlpColumnBlockSize,
//				multIlpRowBlockSize, extraThreads, reductionThreads>(tid,
//				values, columns, rowIndex, vector, result, ilpRows, ilpV,
//				sdata, lastKMRow);
//	} else {
	ilpByRowBase<vSize, vSize, blockSize, ilpColumn, ilpRow,
			multIlpColumnIlpRow, multIlpColumnBlockSize, multIlpRowBlockSize,
			extraThreads, reductionThreads>(tid, values, columns, rowIndex,
			vector, result, sdata);
//	}
}

