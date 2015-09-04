#pragma once

/*
	This class represents a single logical object in the simulation.
*/

#include "Common.h"
#include "Component.h"
#include "Mesh.h"
#include "Transform.h"

#include <string>
#include <vector>

using namespace std;

class Mesh;
class Transform;

class SimObject
{
protected:
	unsigned int m_id;
	string m_name;
	bool m_visible;

	vector<Component*> m_components;

	vector<Mesh*> m_meshes;
	// behaviourComponent collection here
	// collider collection here

	Transform* m_transform;
	// physicalObject here?
public:
	SimObject();
	SimObject(const SimObject*);
	~SimObject();

	unsigned int Initialize(string);
	unsigned int Shutdown();

	unsigned int Update();
	unsigned int Draw();

	void AddComponent(Component*);
	void AddMesh(Mesh*);
	void SetTransform(Transform*);
	void SetVisible(bool);

	void RemoveComponent(Component*);
	void RemoveMesh(Mesh*);
	void RemoveComponent(int);
	void RemoveMesh(int);
	void RemoveTransform();

	Component* GetComponent(unsigned int);
	Mesh* GetMesh(unsigned int);
	Transform* GetTransform();
	bool GetVisible();
	string* GetName();
	unsigned int GetId();
};

