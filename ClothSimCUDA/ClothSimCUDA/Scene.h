#pragma once

/*
	This class is basically a container for all logical objects that take part in the simulation.
	It is an abstract representation. Method Initialize specifies what objects will we load.
*/

#include "Common.h"
#include "SimObject.h"
#include "Camera.h"

#include <string>
#include <vector>

using namespace std;

class Scene
{
protected:
	string m_name;

	vector<SimObject*> m_objects;
	vector<Camera*> m_cameras;
	unsigned int m_currentObjectID;
	unsigned int m_currentCameraID;
public:
	Scene(string);
	Scene(const Scene*);
	~Scene();

	virtual unsigned int Initialize() = 0;
	unsigned int Shutdown();

	unsigned int Update();
	unsigned int Draw();

	unsigned int GetCurrentObjectID();
	unsigned int GetCurrentCameraID();

	void AddObject(SimObject*);
	void AddCamera(Camera*);

	void RemoveObject(SimObject*);
	void RemoveCamera(Camera*);
	void RemoveObject(unsigned int);
	void RemoveCamera(unsigned int);

	SimObject* GetObject();
	Camera* GetCamera();
	SimObject* GetObject(unsigned int);
	Camera* GetCamera(unsigned int);
};

