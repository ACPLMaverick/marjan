#pragma once

/*
	This class is basically a container for all logical objects that take part in the simulation.
*/

#include "Common.h"

#include <string>
#include <vector>

using namespace std;

class Scene
{
private:
	string m_name;
public:
	Scene();
	Scene(const Scene*);
	~Scene();

	unsigned int Initialize();
	unsigned int Shutdown();

	unsigned int Update();
	unsigned int Draw();
};

