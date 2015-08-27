#pragma once

/*
*/

#include "Common.h"

class InputManager
{
protected:
	static InputManager* instance;
	InputManager();
public:
	InputManager(const InputManager*);
	~InputManager();

	static InputManager* GetInstance();
	static void DestroyInstance();

	unsigned int Initialize();
	unsigned int Shutdown();
	unsigned int Run();
};

