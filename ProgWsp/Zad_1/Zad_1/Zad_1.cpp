// Zad_1.cpp : Defines the entry point for the console application.
//

#include<cstdlib>
#include<Windows.h>

#include"Neuron.h"
#include"Network.h"

///GLOBALS///
HANDLE semaphore;
Neuron n1, n2, n3;
Network network;

///METHODS///
DWORD WINAPI NeuronFunc(LPVOID)
{
	return 0;
}

void AppInit()
{
	//Creating semaphore
	semaphore = CreateSemaphore(NULL, 0, 1, L"Przemek");

	//Creating threads
	DWORD ID;
	n1.my_thread = CreateThread(NULL, 0, NeuronFunc, 0, 0, &ID);
	n2.my_thread = CreateThread(NULL, 0, NeuronFunc, 0, 0, &ID);
	n3.my_thread = CreateThread(NULL, 0, NeuronFunc, 0, 0, &ID);
}

///MAIN///
int WinMain(HINSTANCE hInst, HINSTANCE hPrev, LPSTR szCmdLine, int sw)
{
	AppInit();
	network = Network(&n1, &n2, &n3);
	CloseHandle(semaphore);
	for (int i = 0; i < 3; i++)
	{
		TerminateThread(network.neurons[i].my_thread, 0);
		CloseHandle(network.neurons[i].my_thread);
	}
	system("pause");
	return 0;
}

