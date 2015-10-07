#include "Network.h"


Network::Network()
{
}

Network::Network(Neuron* n1, Neuron* n2, Neuron* n3)
{
	neurons[0] = *n1;
	neurons[1] = *n2;
	neurons[2] = *n3;
}

Network::~Network()
{

}
