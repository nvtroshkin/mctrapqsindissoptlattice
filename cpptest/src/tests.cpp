#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "cuda_runtime.h"

#include "definitions.h"

//Tests list
#include "custommathTest.cpp"
#include "ModelTest.cpp"
#include "SolverContextTest.cpp"
#include "SolverTest.cpp"
#include "MonteCarloSimulatorTest.cpp"

int main(int argc, char **argv) {
	printf("Running main() from gtest_main.cc\n");
	testing::InitGoogleTest(&argc, argv);
	std::cout << std::setprecision(12);
	int result = RUN_ALL_TESTS();

	cudaDeviceReset();

	return result;
}
