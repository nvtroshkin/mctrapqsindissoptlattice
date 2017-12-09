#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "definitions.h"

//Tests list
#include "ModelTest.cpp"
#include "SolverTest.cpp"
#include "MonteCarloSimulatorTest.cpp"

int main(int argc, char **argv) {
	printf("Running main() from gtest_main.cc\n");
	testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
