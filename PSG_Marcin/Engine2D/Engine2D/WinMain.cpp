// WinMain.cpp

#include "System.h"

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance,
					PSTR pScmdline, int iCmdshow)
{
	System* system;
	bool result;

	system = new System();
	if (!system) return -1;

	result = system->Initialize();
	if (result) system->Run();

	// shutdown
	system->Shutdown();
	delete system;
	system = nullptr;

	return 0;
}