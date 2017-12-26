/*
 * Timer.cpp
 *
 *  Created on: Dec 26, 2017
 *      Author: fakesci
 */

#include "iostream"
#include "chrono"

#include "Timer.h"

using namespace std::chrono;

void Timer::startCount(const char * title) {
	start = steady_clock::now();
	std::cout << "Starting " << title << "..." << std::endl;
}

void Timer::printElapsedTime(const char * title) {
	auto diff = steady_clock::now() - start;
	std::cout << title << " completed in  "
			<< ((duration<float, seconds::period>) diff).count()
			<< " seconds" << std::endl;
}

