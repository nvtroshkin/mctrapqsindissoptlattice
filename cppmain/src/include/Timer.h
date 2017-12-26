/*
 * Timer.h
 *
 *  Created on: Dec 26, 2017
 *      Author: fakesci
 */

#ifndef SRC_INCLUDE_TIMER_H_
#define SRC_INCLUDE_TIMER_H_

#include "chrono"

class Timer {
	std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
public:
	void startCount(const char * title);
	void printElapsedTime(const char * title);
};


#endif /* SRC_INCLUDE_TIMER_H_ */
