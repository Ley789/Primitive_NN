#pragma once


#include <random>
#include <amp.h>
#include <time.h>
#include <amp_math.h>

template <typename T>
class OutputLayer
{
	concurrency::array_view<T, 1> delta;
	concurrency::array_view<T, 1> output;

	int size;
	void ComputeDelta(std::vector<T> expectedRes);
public:
	OutputLayer();
	~OutputLayer();

	int GetSize();
	void SetOutput(concurrency::array_view<T, 1> &o, int size);
	concurrency::array_view<T, 1>& GetDelta();
	concurrency::array_view<T, 1>& GetOutput();
};

