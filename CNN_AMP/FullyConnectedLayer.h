#pragma once



//TODO RESTRUCT

#include <amp.h>
#include <time.h>
#include <amp_math.h>
#include <random>
//TODO optminze operations (better use of parallelism)

template <typename T>
class FullyConnectedLayer
{
	//the weigths are strored the following way:
	//the column w_ij where i is the reciving unit and j is the sending unit
	concurrency::array_view<T,2> weights;
	concurrency::array<T, 2> upWeights;
	concurrency::array_view<T,1> input;
	concurrency::array_view<T,1> output;
	concurrency::array_view<T, 1> bias;
	concurrency::array_view<T, 1> delta;

	int numRecUnits;
	int numSendUnits;
	void InitRandomWeigths();
	void InitBias();
	inline bool CheckDimension(int s) {
		return s == numSendUnits;
	}

public:
	//TODO init weights but not output
	//w must have r * c elements
	FullyConnectedLayer(int r, int c);
	FullyConnectedLayer(int w[], int r, int c);

	~FullyConnectedLayer() = default;
	void Compute();
	void SetInput(std::vector<T> &val);
	void SetInput(concurrency::array_view<T, 1> &val);
	inline int GetRowSize() {
		return numRecUnits;
	};
	int GetColSize() {
		return numSendUnits;
	};
	concurrency::array_view<T, 1> &GetOutput() {
		return output;
	};
	void ComputeBackwardDelta(const concurrency::array_view<T, 1> &prop);
	void ResetUpdateWeigths();
	void PartialUpdateWeigths(const concurrency::array_view<T, 1> &prop);
};

#pragma region Constructors

template <typename T>
FullyConnectedLayer<T>::FullyConnectedLayer(int w[], int r, int c) : weights(r, c, w), upWeights(r, c),output(r), input(c), bias(r), delta(c) {
	numRecUnits = r;
	numSendUnits = c;
	InitBias();
}

template <typename T>
FullyConnectedLayer<T>::FullyConnectedLayer(int r, int c) : weights(r, c), upWeights(r, c), output(r), input(c), bias(r), delta(c) {
	numRecUnits = r;
	numSendUnits = c;
	InitRandomWeigths();
	InitBias();
}

#pragma endregion

/*
*	Call this after setting the input
*/
template <typename T>
void FullyConnectedLayer<T>::Compute(){
	const concurrency::array_view<T, 2> &w = weights;
	concurrency::array_view<T, 1> &o = output;
	const concurrency::array_view<T, 1> &in = input;
	const concurrency::array_view<T, 1> &b = bias;
	int inner = numSendUnits;

	concurrency::parallel_for_each(
		o.extent,
		[=](concurrency::index<1> idx) restrict(amp) {
		o[idx] = 0;
		int row = idx[0];
		for (int i = 0; i < inner; i++) {
			o[idx] += w(row, i) * in(i);
		}
		o[idx] += b[idx];
		//activation function
		o[idx] = 1 / (1 + (concurrency::fast_math::exp((float)-o[idx])));
	});
}

//now only considers the sigma function
//no optimization in operations transpose multiplication
template <typename T>
void FullyConnectedLayer<T>::ComputeBackwardDelta(const concurrency::array_view<T, 1> &prop) {
	const concurrency::array_view<T, 2> &w = weights;
	concurrency::array_view<T, 2> &d = delta;
	const concurrency::array_view<T, 1> &p = prop;
	const concurrency::array_view<T, 1> &in = input;

	extent<1> e(numSendUnits);
	int inner = numRecUnits;
	concurrency::parallel_for_each(
		e.extent,
		[=](concurrency::index<1> idx) restrict(amp) {
		d[idx] = 0;
		for (int i = 0; i < inner; i++) {
			d[idx] += w[i][idx] * p[i];
		}
		d[idx] = d[idx] * (1 - in[idx]) * in[idx];
	});
}
,

template <typename T>
void ResetUpdateWeigths() {
	concurrency::array<T, 2> &u = upWeights;
	concurrency::parallel_for_each(
		u.extent,
		[=](concurrency::index<2> idx) restrict(amp) {
		int row = idx[0];
		int col = idx[1];
		u[i][j] = 0;
	});
}

/*
	Gets input the delta of next layer
*/
template <typename T>
void PartialUpdateWeigths(const concurrency::array_view<T, 1> &prop) {
	const concurrency::array_view<T, 1> &a = output;

}
/*TODO
template <typename T>
void FullyConnectedLayer<T>::UpdateWeights(float alpha) {
	//TODO make smarter exceptions
	//Move exception to setInput
	if (input.size != numSendUnits) throw 11;

	int inner = numSendUnits;
	concurrency::parallel_for_each(
		output.extent,
		[=](concurrency::index<1> idx) restrict(amp) {
		output[idx] = 0;
		int row = idx[0];
		for (int i = 0; i < inner; i++) {
			output[idx] += weights(row, i) * input(i);
		}
		output[idx] += bias;
		//activation function
		output[idx] = 1 / (1 + (concurrency::fast_math::exp((float)-output[idx]));
	}
	);
}
*/

#pragma region Getter/Setter

template <typename T>
void FullyConnectedLayer<T>::SetInput(std::vector<T> &val) {
	concurrency::array<T,1> i(val.size(), val.begin(), val.end());
	i.copy_to(input);
}

template <typename T>
void FullyConnectedLayer<T>::SetInput(concurrency::array_view<T, 1> &val) {
	val.copy_to(input);
}
#pragma endregion

#pragma region Utility
//TODO test resulting values
template <typename T>
void FullyConnectedLayer<T>::InitRandomWeigths() {
	std::random_device rd;
	std::mt19937 gen(rd());

	std::normal_distribution<> d(0, 0.2);
	for (int i = 0; i < numRecUnits; i++) {
		for (int j = 0; j < numSendUnits; j++) {
			double val = (double)rand();
			weights[i][j] = (T)d(gen);
		}
	}
}

template <typename T>
void FullyConnectedLayer<T>::InitBias() {
	for (int i = 0; i < numRecUnits; i++) {
			bias[i] = (T)1;
	}
}

#pragma endregion

