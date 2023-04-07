#pragma once
#define _USE_MATH_DEFINES
#include <vector>
#include <math.h>

using namespace std;

class FFT
{
public:
	vector<double> real;
	vector<double> imag;
public:
	FFT() = default;
	void process(vector<double>& signal);
};

