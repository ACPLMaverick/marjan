#pragma once
#include <Windows.h>
class Neuron
{
public:
	HANDLE my_thread;
	int activation;
	int weights[3];

	Neuron();
	Neuron(int* w);
	~Neuron();
};

