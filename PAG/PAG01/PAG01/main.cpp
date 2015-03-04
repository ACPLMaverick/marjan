#include <Windows.h>

#include "System.h"

int main()
{
	System* system = new System();
	system->Initialize();
	system->GameLoop();
	system->Shutdown();
	delete system;
	return 0;
}