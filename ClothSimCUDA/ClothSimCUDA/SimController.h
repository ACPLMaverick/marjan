#pragma once

/*
	This component is the main controller of the whole simulation. Mainly responds to user input.
*/

#include "Common.h"
#include "Component.h"
#include "InputHandler.h"

class SimController :
	public Component
{
public:
	SimController(SimObject* obj);
	SimController(const SimController*);
	~SimController();

	virtual unsigned int Initialize();
	virtual unsigned int Shutdown();

	virtual unsigned int Update();
	virtual unsigned int Draw();
};

