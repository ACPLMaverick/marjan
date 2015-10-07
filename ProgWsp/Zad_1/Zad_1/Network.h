#pragma once

#include "Neuron.h"

class Network
{
public:
	Neuron neurons[3];
	int output_value[3];

	Network();
	Network(Neuron*, Neuron*, Neuron*);
	~Network();
};

