#pragma once

#include <amp.h>


template <typename T>
class OutputLayer
{
	concurrency::array_view<T, 1> delta;
	const concurrency::array_view<T, 1> &output;

	int size;
	void ComputeDelta(std::vector<T> &expectedRes);

public:
	OutputLayer(const concurrency::array_view<T, 1> &o, int s) : output(o), delta(s){
		size = s;
	}
	~OutputLayer() = default;

	int GetSize() {
		return size;
	}

	concurrency::array_view<T, 1>& GetDelta() {
		return delta;
	}
	const concurrency::array_view<T, 1>& GetOutput() {
		return output;
	}
};

/*
	Assumes that computation have already been done.
	Assumes sigmund activation function.
*/
template <typename T>
void OutputLayer<T>::ComputeDelta(std::vector<T> &expectedRes) {
	//Fix exception
	if (size != expectedRes.size()) throw 11;
	concurrency::array<T, 1> res(expectedRes.size(), expectedRes.begin(), expectedRes.end());

	res.copy_to(delta);

	concurrency::array_view<T, 1> &d = delta;
	const concurrency::array_view<T, 1> &o = output;

	concurrency::parallel_for_each(
		res.extent,
		[=](concurrency::index<1> idx) restrict(amp) {
			d[idx] = (-1) * (d[idx] - o[idx]) * (1 - o[idx]) * o[idx];
	});
}