#pragma once

/*
	Component.
*/

#include "SimObject.h"

class Component
{
protected:
	SimObject* m_obj;
public:
	Component(SimObject* obj);
	Component(const Component*);
	~Component();

	virtual unsigned int Initialize() = 0;
	virtual unsigned int Shutdown() = 0;

	virtual unsigned int Update() = 0;
	virtual unsigned int Draw() = 0;

	virtual SimObject* GetMySimObject() final;
};

