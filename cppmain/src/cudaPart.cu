/*
 * cudaPart.cu
 *
 *  Created on: Dec 19, 2017
 *      Author: fakesci
 */

//Compute capability 1.x doesn't allow separate compilation
#include "precision-definition.h"

#include "Solver.cpp"

#include "custommath.cpp"

#include "helper_cuda.h"

#ifdef TEST_MODE
#include "../../cpptest/src/SolverTest0.cpp"
#include "../../cpptest/src/custommathTest0.cpp"
#endif

__global__ void simulateKernel(Solver * const * const solverDevPtrs) {
	solverDevPtrs[blockIdx.x]->solve();
}

void simulate(const uint nBlocks, const uint nThreadsPerBlock,
		Solver * const * const solverDevPtrs) {
	simulateKernel<<<nBlocks, nThreadsPerBlock>>>(solverDevPtrs);

	//check for the kernel errors
	getLastCudaError("Simulation failed");
}
