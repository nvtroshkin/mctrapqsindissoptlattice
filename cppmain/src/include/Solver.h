/*
 * Solver.h
 *
 *  Created on: Nov 24, 2017
 *      Author: fakesci
 */

#ifndef SRC_INCLUDE_SOLVER_H_
#define SRC_INCLUDE_SOLVER_H_

#include <precision-definition.h>

#include <curand_kernel.h>

class Solver {

	//------------------constants--------------------------
	const FLOAT_TYPE tStep;

	const int nTimeSteps;

	//------------------model------------------------------
	const int basisSize;

#ifdef L_SPARSE
	__restrict__ const CSR3Matrix * const lCSR3;
#else
	__restrict__ const CUDA_COMPLEX_TYPE * const rungeKuttaOperator;
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
	curandStateMRG32k3a_t randomGeneratorState;

	// shared via block mutables
	FLOAT_TYPE * const sharedNormThresholdPtr;

	FLOAT_TYPE * const sharedFloatPtr;

	CUDA_COMPLEX_TYPE ** const sharedPointerPtr;

	CUDA_COMPLEX_TYPE * const sharedK1;

	CUDA_COMPLEX_TYPE * const sharedK2;

	CUDA_COMPLEX_TYPE * const sharedK3;

	CUDA_COMPLEX_TYPE * const sharedK4;

	CUDA_COMPLEX_TYPE * sharedPrevState;

	CUDA_COMPLEX_TYPE * sharedCurState;

//------------------Declarations-----------------------

	__device__
	void parallelCalcCurrentY();

	__device__
	void parallelMultLV(const CUDA_COMPLEX_TYPE * const vector,
	CUDA_COMPLEX_TYPE *result);

	__device__
	void normalizeVector(CUDA_COMPLEX_TYPE *stateVector,
			const CUDA_COMPLEX_TYPE &stateVectorNorm2,
			CUDA_COMPLEX_TYPE *result);

	__device__ FLOAT_TYPE calcNormSquare(const CUDA_COMPLEX_TYPE * const v);

	__device__
	void multiplyRow(uint rowsize, const int * const columnIndices,
			const CUDA_COMPLEX_TYPE * const matrixValues,
			const CUDA_COMPLEX_TYPE * const vector,
			CUDA_COMPLEX_TYPE &result);

	__device__
	void printLog(const char * str);

	__device__ FLOAT_TYPE getNextRandomFloat();

public:
	__host__ __device__
	Solver(int basisSize, FLOAT_TYPE timeStep, int nTimeSteps,
			const CUDA_COMPLEX_TYPE * rungeKuttaOperator, int a1CSR3RowsNum,
			const CUDA_COMPLEX_TYPE * a1CSR3Values, const int * a1CSR3Columns,
			const int * a1CSR3RowIndex, int a2CSR3RowsNum,
			const CUDA_COMPLEX_TYPE * a2CSR3Values, const int * a2CSR3Columns,
			const int * a2CSR3RowIndex, int a3CSR3RowsNum,
			const CUDA_COMPLEX_TYPE * a3CSR3Values, const int * a3CSR3Columns,
			const int * a3CSR3RowIndex,
			//non-const
			FLOAT_TYPE * sharedNormThresholdPtr,
			FLOAT_TYPE * sharedFloatPtr,
			CUDA_COMPLEX_TYPE ** sharedPointerPtr,
			CUDA_COMPLEX_TYPE *sharedK1, CUDA_COMPLEX_TYPE *sharedK2,
			CUDA_COMPLEX_TYPE *sharedK3, CUDA_COMPLEX_TYPE *sharedK4,
			CUDA_COMPLEX_TYPE *sharedPrevState,
			CUDA_COMPLEX_TYPE *sharedCurState
			);

	/**
	 * Stores the final result in the curStep
	 */
	__device__
	void solve();

	__device__
	void parallelNormalizeVector(CUDA_COMPLEX_TYPE *stateVector);

	__device__
	void parallelMultMatrixVector(const CUDA_COMPLEX_TYPE * const matrix,
			const int rows, const int columns,
			const CUDA_COMPLEX_TYPE * const vector,
			CUDA_COMPLEX_TYPE * const result);

	__device__
	void parallelMultCSR3MatrixVector(const CUDA_COMPLEX_TYPE * const csr3MatrixValues,
			const int * const csr3MatrixColumns,
			const int * const csr3MatrixRowIndex,
			const CUDA_COMPLEX_TYPE * const vector, CUDA_COMPLEX_TYPE *result);

	__device__
	void parallelCopy(const CUDA_COMPLEX_TYPE * const source,
	CUDA_COMPLEX_TYPE * const dest);

	__device__
	void parallelCalcAlphaVector(const FLOAT_TYPE alpha,
			const CUDA_COMPLEX_TYPE * const vector,
			CUDA_COMPLEX_TYPE * const result);

	__device__
	void parallelCalcV1PlusAlphaV2(const CUDA_COMPLEX_TYPE * const v1,
			const FLOAT_TYPE alpha, const CUDA_COMPLEX_TYPE * const v2,
			CUDA_COMPLEX_TYPE * const result);

	__device__
	void parallelRungeKuttaStep();

	__device__
	void parallelMakeJump();

	//-----------------Getters-----------------------------

	const CUDA_COMPLEX_TYPE * getCurState();
};

inline const CUDA_COMPLEX_TYPE * Solver::getCurState() {
	return sharedCurState;
}

#endif /* SRC_INCLUDE_SOLVER_H_ */
