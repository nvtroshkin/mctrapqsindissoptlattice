/*
 * macro.h
 *
 *  Created on: Nov 22, 2017
 *      Author: fakesci
 */

#ifndef SRC_INCLUDE_PRECISION_DEFINITION_H_
#define SRC_INCLUDE_PRECISION_DEFINITION_H_

#include <cuComplex.h>
#define MKL_Complex16 cuDoubleComplex
#define MKL_Complex8 cuFloatComplex

#include <mkl.h>
#include <ipps.h>

//used precision - uncomment the right row
#define SINGLE_PRECISION
//#define DOUBLE_PRECISION

//definitions for single precision
#ifdef SINGLE_PRECISION
#define FLOAT_TYPE float
#define COMPLEX_TYPE MKL_Complex8

//mkl functions
#define complex_cblas_copy(...) cblas_ccopy (__VA_ARGS__)
#define complex_vMul(...) vcMul (__VA_ARGS__)
#define complex_cblas_dotc_sub(...) cblas_cdotc_sub (__VA_ARGS__)
#define complex_mkl_cspblas_csrgemv(...) mkl_cspblas_ccsrgemv (__VA_ARGS__)
#define complex_mkl_cblas_gemv(...) cblas_cgemv (__VA_ARGS__)
#define complex_cblas_axpy(...) cblas_caxpy (__VA_ARGS__)
#define complex_mkl_dnscsr(...) mkl_cdnscsr (__VA_ARGS__)
#define complex_cblas_nrm2(...) cblas_scnrm2 (__VA_ARGS__)
#define complex_cblas_gemm(...) cblas_cgemm (__VA_ARGS__)
#define complex_cblas_scal(...) cblas_csscal (__VA_ARGS__)

#define ippsSum_f(...) ippsSum_32f (__VA_ARGS__, ippAlgHintAccurate)
#define cblas_dot(...) cblas_dsdot (__VA_ARGS__)
#define cblas_copy(...) cblas_scopy (__VA_ARGS__)
#define cblas_scal(...) cblas_sscal (__VA_ARGS__)
#define vSqrt(...) vsSqrt (__VA_ARGS__)
#define vRngUniform(...) vsRngUniform (__VA_ARGS__)

//GPU
#define CUDA_COMPLEX_TYPE cuFloatComplex
#define rsqrt(...) rsqrtf (__VA_ARGS__)
#define cuSqrt(...) sqrtf (__VA_ARGS__)

#endif

//definitions for double precision
#ifdef DOUBLE_PRECISION
#define FLOAT_TYPE double
#define COMPLEX_TYPE MKL_Complex16
#define CUDA_COMPLEX cuDoubleComplex

//mkl functions
#define complex_cblas_copy(...) cblas_zcopy (__VA_ARGS__)
#define complex_vMul(...) vzMul (__VA_ARGS__)
#define complex_cblas_dotc_sub(...) cblas_zdotc_sub (__VA_ARGS__)
#define complex_mkl_cspblas_csrgemv(...) mkl_cspblas_zcsrgemv (__VA_ARGS__)
#define complex_mkl_cblas_gemv(...) cblas_zgemv (__VA_ARGS__)
#define complex_cblas_axpy(...) cblas_zaxpy (__VA_ARGS__)
#define complex_mkl_dnscsr(...) mkl_zdnscsr (__VA_ARGS__)
#define complex_cblas_nrm2(...) cblas_dznrm2 (__VA_ARGS__)
#define complex_cblas_gemm(...) cblas_zgemm (__VA_ARGS__)
#define complex_cblas_scal(...) cblas_zdscal (__VA_ARGS__)

#define ippsSum_f(...) ippsSum_64f (__VA_ARGS__)
#define cblas_dot(...) cblas_ddot (__VA_ARGS__)
#define cblas_copy(...) cblas_dcopy (__VA_ARGS__)
#define cblas_scal(...) cblas_dscal (__VA_ARGS__)
#define vSqrt(...) vdSqrt (__VA_ARGS__)
#define vRngUniform(...) vdRngUniform (__VA_ARGS__)

//GPU
#define CUDA_COMPLEX_TYPE cuDoubleComplex
#define curand_uniform(...) curand_uniform_double (__VA_ARGS__)
#define cuSqrt(...) sqrt (__VA_ARGS__)
#endif

#endif /* SRC_INCLUDE_PRECISION_DEFINITION_H_ */
