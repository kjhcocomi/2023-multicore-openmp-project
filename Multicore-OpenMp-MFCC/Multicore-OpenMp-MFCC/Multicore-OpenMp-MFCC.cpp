#include <iostream>
#include "MFCC.h"

int main()
{
	MFCC mfcc = MFCC();
	vector<double> test = vector<double>(1025, 10);
	vector<vector<double>> res = mfcc.dctMfcc(test);
	for (auto a : res) {
		for (auto b : a) {
			cout << fixed;
			cout.precision(5);
			cout << b << "   ";
		}
		cout << "\n";
	}
}