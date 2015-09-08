#pragma once

/*
This class represents an abstraction of a single drawable mesh.
*/

#include "Common.h"
#include "SimObject.h"
#include "Component.h"

#include <vector>

class SimObject;
class Component;

class Mesh : public Component
{
protected:

	// collection of textures?
	virtual void GenerateVertexData() = 0;
public:
	Mesh(SimObject* obj);
	Mesh(const Mesh*);
	~Mesh();

	virtual unsigned int Initialize() = 0;
	virtual unsigned int Shutdown() = 0;

	virtual unsigned int Update() final;
	virtual unsigned int Draw() = 0;
};

