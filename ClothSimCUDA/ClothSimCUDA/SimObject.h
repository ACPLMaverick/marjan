#pragma once

/*
	This class represents a single logical object in the simulation.
*/

#include "Common.h"

#include <string>
#include <vector>

using namespace std;

class SimObject
{
protected:
	unsigned int id;
	string name;

	// mesh collection here
	// behaviourComponent collection here
	// collider collection here
	// transform here
	// physicalObject here?
public:
	SimObject();
	SimObject(const SimObject*);
	~SimObject();

	unsigned int Initialize();
	unsigned int Shutdown();

	unsigned int Update();
	unsigned int Draw();
};

