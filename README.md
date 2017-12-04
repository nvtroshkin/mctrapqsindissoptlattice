# Description

The project aims to repeat the results by H.J. Carmichael in his paper ["Breakdown of Photon Blockade: A Dissipative Quantum Phase Transition in Zero Dimensions", Phys. Rev. X 5, 031028 (2015)](https://journals.aps.org/prx/abstract/10.1103/PhysRevX.5.031028) by a Monte-Carlo simulations algorithm described in a book by H.-P. Breuer and F. Petruccione ["The Theory of Open Quantum Systems", Oxford University Press, 2002, p.364](https://www.researchgate.net/publication/235426843_The_Theory_of_Open_Quantum_Systems).

A cavity with a Jaynes-Cummings non-linearity, coherent classical drive and dissipation is studied by Monte-Carlo simulations in the dressed states representation. 

It is assumed that initially the system is in it's ground state, then N different realiztions of a partly deterministic random process are generated and, finally, the mean photon number and standard deviation of photon numbers are calculated.

# Requirements

1. [Intel Math Kernel Library](https://software.intel.com/en-us/mkl)
2. [Intel Threading Building Blocks](https://software.intel.com/en-us/intel-tbb)
3. [Google Test](https://github.com/google/googletest) for testing 
4. OpenMP libraries

# Installation in Eclipse IDE

There are two projects embedded into one repository: OneAtom - source code, OneAtomTest - tests.

## OneAtom project configuration

1. Create C++ prject from source at ./OneAtom
1. Project > Properties > C/C++ Build > Settings > C++ Compiler > Includes > Add
    1. <Intel TBB base dir>/include/tbb
    1. <Intel MKL base dir>/mkl/include
    1. ./src/include
1. Ibid > Miscellaneous > Other flags > Add -fopenmp
1. ... > C++ Linker > Libraries > add
    1. mkl_intel_lp64
    1. mkl_core
    1. tbbmalloc
    1. mkl_sequential
1. Ibid at the bottom > Library search path > Add
  1. <Intel MKL base dir>/mkl/lib/intel64
  1. <Intel TBB base dir>/lib/intel64/<your compiler folder>
1. Ibid > Miscellaneous > Linker flags > Add -fopenmp
1. Run > Run Configurations > C/C++ Application > OneAtom Debug > Environment > New
  1. Linux: LD_LIBRARY_PATH = "<Intel MKL base dir>/mkl/lib/intel64_lin/:<Intel TBB base dir>/lib/intel64/gcc4.7/"
  1. Windows: PATH = ...

## OneAtomTest project configuration

1. Create C++ prject from source at ./OneAtom
1. If there is no imported_src source directory linked to the OneAtom project, create it
  1. Project > New > Folder > Advanced > Linked Folder
  1. Choose the src folder form the OneTest project
1. Project > Properties > C/C++ Build > Settings > C++ Compiler > Includes > Add
  1. <Intel TBB base dir>/include/tbb
  1. <Intel MKL base dir>/mkl/include
  1. OneAtom/src/include - header files from the OneTest project
  1. <googletest release dir>/googletest/include
  1. <googletest release dir>/googlemock/include
1. Ibid > Miscellaneous > Other flags > Add -fopenmp
1. ... > C++ Linker > Libraries > add 
  1. mkl_intel_lp64
  1. mkl_core
  1. tbbmalloc
  1. mkl_sequential
  1. pthread
1. Ibid at the bottom > Library search path > Add
  1. <Intel MKL base dir>/mkl/lib/intel64
  1. <Intel TBB base dir>/lib/intel64/<your compiler folder>
1. Ibid > Miscellaneous >
  1. > Linker flags > Add -fopenmp
  1. > Other objects > Add
    1. <googletest release dir>/googletest/make/gtest_main.a
    1. <googletest release dir>/googlemock/make/gmock_main.a
1. Run > Run Configurations > C/C++ Unit > OneAtomTest Debug > Environment > New
  1. Linux: LD_LIBRARY_PATH = "<Intel MKL base dir>/mkl/lib/intel64_lin/:<Intel TBB base dir>/lib/intel64/gcc4.7/"
  1. Windows: PATH = ...

# Usage

## Main Project (OneAtom)

All settings are scattered among two main header files:
- "eval-params.h" contains all parameters dealing with the numerical method: steps number, basis size, etc.
- "system-constants.h" contains all physical parameters like driving field strength

The project uses OpenMP for parallelization. 

Be sure, that you have enough random numbers in buffers or it will crash during calculations with an exception.

## Tests (OneAtomTest)

Just launch to ensure that all is OK.

WARNING: Tests work with double precision only.
