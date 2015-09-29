#pragma once

/*
A simple sphere collider.
*/

#include "Collider.h"

#include <glm\glm\glm.hpp>

class Collider;

class SphereCollider :
	public Collider
{
protected:
	glm::vec3 m_offset;
	float m_radius;
public:
	SphereCollider(SimObject*);
	SphereCollider(SimObject*, glm::vec3*, float);
	~SphereCollider();

	virtual unsigned int Initialize();
	virtual unsigned int Shutdown();

	virtual unsigned int Update();
	virtual unsigned int Draw();

	virtual CollisonTestResult TestWithBoxAA(BoxAACollider*);
	virtual CollisonTestResult TestWithSphere(SphereCollider*);
};

