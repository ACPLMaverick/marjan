#pragma once

/*
This class manages all collisions that take place in the scene.
It is also responsible for solving forces and object movements.
*/

#include "Singleton.h"
#include "Collider.h"

#include <vector>

class Collider;

class PhysicsManager :
	public Singleton<PhysicsManager>
{
	friend class Singleton<PhysicsManager>;
private:
	std::vector<Collider*> m_colliders;

	float m_gravity;
	bool m_ifDrawColliders;

	PhysicsManager();
public:
	PhysicsManager(const PhysicsManager*);
	~PhysicsManager();

	unsigned int Initialize();
	unsigned int Shutdown();
	unsigned int Run();

	void AddCollider(Collider*);
	bool RemoveCollider(Collider*);

	float GetGravity();
	bool GetIfDrawColliders();
};

