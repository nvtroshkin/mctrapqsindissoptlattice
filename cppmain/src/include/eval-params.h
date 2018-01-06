/*
 *  Created on: Nov 18, 2017
 *      Author: fakesci
 */

#ifndef SRC_EVAL_PARAMS_H_
#define SRC_EVAL_PARAMS_H_

#include <precision-definition.h>

/**
 * If defined, progress of calculations is printed every N samples
 */
#define PRINT_PROGRESS
static const int MIN_SAMPLES_BETWEEN_PROGRESS = 50;

/**
 * Consider L as a sparse matrix
 * (don't know whether useful or not, may be for bigger dimensions)
 */
//#define L_SPARSE

/**
 * Print info about how dense matrices of the operators are
 */
//#define CHECK_SPARSITY

/**
 * Change to adjust the number of threads per block. All the threads are divided on warps
 * that execute independently. More threads - less resources per thread available
 *
 * Please, keep in mind that it is very GPU-specific. My GPU's multiprocessors can
 * hold only 1 warp (32 threads) at once, others can more (one thread - one core).
 * Also one warp may be not sufficient to get 100% load.
 *
 * It is better to have it is a factor of the basis size (or is could be slowed down by factor of 10),
 * because matrices and vectors  * are processed by tiles (32 threads per warp and 8 cores per SM - should be multiple
 * of it too)
 */
static const uint CUDA_N_WARPS_PER_BLOCK = 1;

/**
 * Don't change it. The standard warp size on GPUs
 */
static const uint CUDA_WARP_SIZE = 32;

/**
 * Don't change it. The total number of threads. To adjust use nWarpsPerBlock constant
 */
static constexpr uint CUDA_THREADS_PER_BLOCK = CUDA_WARP_SIZE * CUDA_N_WARPS_PER_BLOCK;

/**
 * Defines the ILP (instruction level parallelism) level of the density matrix - vector multiplication code.
 * More ILP - more registers needed, more cycles are unrolled, more code with all consequences (Instruction cash thrashing,
 * though never noticed). PTXAS optimizations makes large level of ILP worthless because it divides global loads
 * with arithmetic. So 2 probably is a reasonable maximum.
 *
 * If ILP_COLUMN = 1, then each thread gets one column, multiplies it on a vector element, stores it in the shared memory.
 * If there are elements left (basis is larger then threads number), it gets another column and etc.
 *
 * If ILP_COLUMN = 2, it loads 2 columns before arithmetics (if there are sufficient amount of columns)
 *
 * It is better to choose the value in such a way that basisSize divides on threadsNumber*ILP_COLUMN or performance could
 * degrade
 */
static const uint CUDA_MATRIX_VECTOR_ILP_COLUMN = 1;

/**
 * Same as ILP_COLUMN but by rows. Also it saves global loads of vector values (several row values are multiplied on on vector element)
 *
 * The basisSize should divide on ILP_ROW or performance could degrade
 */
static const uint CUDA_MATRIX_VECTOR_ILP_ROW = 1;

/**
 * Same, but for sparse matrices (jumps). Note that sparse matrix multiplications are not very expensive.
 * They have approx. n=basisSize non-zero elements less in a row then dense matrices.
 */
static const uint CUDA_SPARSE_MATRIX_VECTOR_ILP_COLUMN = 3;

/**
 * One block = one trajectory. If it is small - there are many separate kernel
 * invocations with corresponding start up expenses. If there are many - fewer resources
 * are available for each thread on the GPU.
 *
 * It is better to have it as a factor of the samples number and as a multiple of SM's
 * number
 */
static const uint CUDA_N_BLOCKS = 32;

//Evaluation of each sample is performed beginning at 0s and ending at the end time.
//Increasing the END_TIME value is necessary to caught the stationary evaluation
//phase
static const FLOAT_TYPE TIME_STEP_SIZE = 0.00001;
static const int TIME_STEPS_NUMBER = 1000;		//the total number of steps
static const int MONTE_CARLO_SAMPLES_NUMBER = 128;

static const FLOAT_TYPE EVAL_TIME = TIME_STEP_SIZE * TIME_STEPS_NUMBER;

//----------------debug info--------------------------------

/**
 * Defines how often debug info is printed
 */
static const int TIME_STEPS_BETWEEN_DEBUG = 1000;

/**
 * If defined, debug info relating to the continuous evolution is printed every
 * N steps of the solver
 */
//#define DEBUG_CONTINUOUS

/**
 * If defined, debug info related to random jumps is printed.
 * Some of it is printed when a jump occurred, some - every N steps
 */
//#define DEBUG_JUMPS

//----------------Tests-----------------------------------------

/**
 * Changes code so it could be easily tested - pass as a parameter or define in tests
 */
//#define TEST_MODE

#endif /* SRC_EVAL_PARAMS_H_ */
