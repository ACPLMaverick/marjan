#pragma once

/*
*/

#include "Common.h"
#include "InputManager.h"

class InputHandler
{
protected:
	static InputHandler* instance;
	InputHandler();
public:
	InputHandler(const InputHandler*);
	~InputHandler();

	static InputHandler* GetInstance();
	static void DestroyInstance();


	bool ExitPressed();
};

