/*
 * cudaPart.cu
 *
 *  Created on: Dec 19, 2017
 *      Author: fakesci
 */

//Compute capability 1.x doesn't allow separate compilation
#include "Solver.cpp"
#include "MonteCarloSimulator.cpp"

#ifdef TEST_MODE
#include "../../cpptest/src/SolverTest0.cpp"
#endif


