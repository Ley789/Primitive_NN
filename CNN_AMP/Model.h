#pragma once
#include "FullyConnectedLayer.h"
#include "OutputLayer.h"
#include <vector>
#include <memory>

class Model
{
	int inputSize;
	std::vector< std::shared_ptr< FullyConnectedLayer<float>>> model;
	OutputLayer<float> output;

	void ComputeHyp(std::vector<float> input);
	bool done;

public:
	Model(int size) {
		inputSize = size;
		done = false;
	}
	~Model();
	void AddFullNetworkLayer(int outputSize);
	void RemoveLastLayer();
	void Finish();
	std::vector<float> GetHyp(std::vector<float> input);
};
