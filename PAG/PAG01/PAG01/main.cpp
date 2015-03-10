#include <Windows.h>

#include "System.h"

int main()
{
	System* system = System::GetInstance();
	system->Initialize();
	system->GameLoop();
	system->Shutdown();
	System::DestroyInstance();
	return 0;
}