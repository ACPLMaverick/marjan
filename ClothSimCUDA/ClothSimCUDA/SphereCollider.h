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
	friend class BoxAACollider;
protected:
	glm::vec3 m_center;
	glm::vec3 m_effectiveCenter;
	float m_radius;
	float m_effectiveRadius;
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

