/*
 * Solver.h
 *
 *  Created on: Nov 24, 2017
 *      Author: fakesci
 */

#ifndef SRC_INCLUDE_SOLVER_H_
#define SRC_INCLUDE_SOLVER_H_

#include <precision-definition.h>
#include <iostream>
#include <CSR3Matrix.h>
#include <Model.h>

#include <cublas_v2.h>
#include <curand_kernel.h>

class Solver {

	//------------------constants--------------------------
	const FLOAT_TYPE tStep;

	const FLOAT_TYPE tHalfStep;

	const FLOAT_TYPE tSixthStep;

	const int ntimeSteps;

	//------------------model------------------------------
	const int basisSize;

#ifdef L_SPARSE
	__restrict__ const CSR3Matrix * const lCSR3;
#else
	__restrict__                             const CUDA_COMPLEX_TYPE * const l;
#endif

	const int a1CSR3RowsNum;
	const CUDA_COMPLEX_TYPE * __restrict__ const a1CSR3Values;
	const int * __restrict__ const a1CSR3Columns;
	const int * __restrict__ const a1CSR3RowIndex;

	const int a2CSR3RowsNum;
	const CUDA_COMPLEX_TYPE * __restrict__ const a2CSR3Values;
	const int * __restrict__ const a2CSR3Columns;
	const int * __restrict__ const a2CSR3RowIndex;

	const int a3CSR3RowsNum;
	const CUDA_COMPLEX_TYPE * __restrict__ const a3CSR3Values;
	const int * __restrict__ const a3CSR3Columns;
	const int * __restrict__ const a3CSR3RowIndex;

	//------------------caches-----------------------------
	curandStateMRG32k3a_t state;

	//shared between threads in a block
	FLOAT_TYPE * __restrict__ const svNormThresholdPtr;

	FLOAT_TYPE * __restrict__ const sharedFloatPtr;

	CUDA_COMPLEX_TYPE ** sharedPointerPtr;

	CUDA_COMPLEX_TYPE * __restrict__ const k1;

	CUDA_COMPLEX_TYPE * __restrict__ const k2;

	CUDA_COMPLEX_TYPE * __restrict__ const k3;

	CUDA_COMPLEX_TYPE * __restrict__ const k4;

	CUDA_COMPLEX_TYPE * __restrict__ prevState;

	CUDA_COMPLEX_TYPE * __restrict__ curState;

	//----------------------------------------------------

#if defined(DEBUG_CONTINUOUS) || defined(DEBUG_JUMPS)
	bool shouldPrintDebugInfo = false;
#endif

//------------------Declarations-----------------------

	__device__
	void make4thOrderRungeKuttaStep();

	__device__
	void normalizeVector(CUDA_COMPLEX_TYPE *stateVector,
			const CUDA_COMPLEX_TYPE &stateVectorNorm2,
			CUDA_COMPLEX_TYPE *result);

	__device__
	void parallelMultLV(CUDA_COMPLEX_TYPE *vector,
	CUDA_COMPLEX_TYPE *result);

	__device__
	void parallelMultMatrixVector(const CUDA_COMPLEX_TYPE * const matrix,
			const int &rows, const int &columns,
			CUDA_COMPLEX_TYPE *vector,
			CUDA_COMPLEX_TYPE *result);

	__device__
	void parallelMultCSR3MatrixVector(const int csr3MatrixRowsNum,
			const CUDA_COMPLEX_TYPE * const csr3MatrixValues,
			const int * const csr3MatrixColumns,
			const int * const csr3MatrixRowIndex,
			CUDA_COMPLEX_TYPE *vector, CUDA_COMPLEX_TYPE *result);

	__device__ FLOAT_TYPE calcNormSquare(CUDA_COMPLEX_TYPE * v);

	__device__
	void parallelCopy(CUDA_COMPLEX_TYPE * source,
	CUDA_COMPLEX_TYPE * dest);

	__device__
	void parallelCalcAlphaVector(const FLOAT_TYPE &alpha,
	CUDA_COMPLEX_TYPE * vector, CUDA_COMPLEX_TYPE * result);

	__device__
	void parallelCalcV1PlusAlphaV2(
	CUDA_COMPLEX_TYPE * v1, const FLOAT_TYPE &alpha, CUDA_COMPLEX_TYPE * v2,
	CUDA_COMPLEX_TYPE * result);

	__device__
	void parallelCalcCurrentY();

	__device__ CUDA_COMPLEX_TYPE multiplyRow(uint rowsize,
			const int * const columnIndices,
			const CUDA_COMPLEX_TYPE * const matrixValues,
			const CUDA_COMPLEX_TYPE * const vector);

public:
	__device__
	Solver(int basisSize, FLOAT_TYPE timeStep, int timeStepsNumber,
	CUDA_COMPLEX_TYPE * l, int a1CSR3RowsNum,
	CUDA_COMPLEX_TYPE * a1CSR3Values, int * a1CSR3Columns, int * a1CSR3RowIndex,
			int a2CSR3RowsNum,
			CUDA_COMPLEX_TYPE * a2CSR3Values, int * a2CSR3Columns,
			int * a2CSR3RowIndex, int a3CSR3RowsNum,
			CUDA_COMPLEX_TYPE * a3CSR3Values, int * a3CSR3Columns,
			int * a3CSR3RowIndex,
			FLOAT_TYPE * svNormThresholdPtr,
			FLOAT_TYPE * sharedFloatPtr, CUDA_COMPLEX_TYPE ** sharedPointerPtr,
			CUDA_COMPLEX_TYPE *k1,
			CUDA_COMPLEX_TYPE *k2,
			CUDA_COMPLEX_TYPE *k3, CUDA_COMPLEX_TYPE *k4,
			CUDA_COMPLEX_TYPE *prevState, CUDA_COMPLEX_TYPE *curState);

	/**
	 * Stores the final result in the curStep
	 */
	__device__
	void solve();

	__device__
	void parallelNormalizeVector(CUDA_COMPLEX_TYPE *stateVector);

	__device__
	void makeJump();
};

#endif /* SRC_INCLUDE_SOLVER_H_ */
