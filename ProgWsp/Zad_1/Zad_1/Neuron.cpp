#include "Neuron.h"

Neuron::Neuron()
{

}

Neuron::Neuron(int* w)
{
	for (int i = 0; i < 3; i++)
	{
		weights[i] = *(w + i);
	}
}

Neuron::~Neuron()
{

}


