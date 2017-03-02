#include "Model.h"

Model::~Model(){
}

void Model::AddFullNetworkLayer(int outPutSize) {
	if(model.size() == 0){
		std::shared_ptr< FullyConnectedLayer<float> > p = std::make_shared< FullyConnectedLayer<float>> (outPutSize, inputSize);
		model.push_back(p);
	}
	else {
		std::shared_ptr< FullyConnectedLayer<float> > p = std::make_shared< FullyConnectedLayer<float>>(outPutSize, model.back()->GetRowSize());
		model.push_back(p);
	}
}

void Model::RemoveLastLayer() {
	model.pop_back();
}

void Model::ComputeHyp(std::vector<float> input) {
	//fix exception
	if (!done) throw 10;
	if (input.size() != inputSize) throw 11;

	model[0]->SetInput(input);
	model[0]->Compute();
	for (unsigned i = 1; i < model.size(); i++) {
		model[i]->SetInput(model[i-1]->GetOutput());
		model[i]->Compute();
	}
}

//fix exception
void Model::Finish() {
	if (model.size() < 0) throw 12;
	auto p = model.back();
	output = std::make_shared< OutputLayer<float>>(p->GetOutput(), p->GetRowSize());
	done = true;
}

std::vector<float> Model::GetHyp(std::vector<float> input) {
	std::vector<float> res;

	ComputeHyp(input);

	auto val = output->GetOutput();
	for (int i = 0; i < output->GetSize(); i++) {
		res.push_back(val(i));
	}
	return res;
}