#pragma once

/*
*/

#include "Collider.h"
#include "MeshGLPlane.h"

class Collider;

class ClothCollider :
	public Collider
{
protected:
	MeshGLPlane* m_meshPlane;
public:
	ClothCollider(SimObject* obj);
	ClothCollider(const ClothCollider*);
	~ClothCollider();

	virtual unsigned int Initialize();
	virtual unsigned int Shutdown();

	virtual unsigned int Update();
	virtual unsigned int Draw();

	virtual CollisonTestResult TestWithBoxAA(BoxAACollider* other);
	virtual CollisonTestResult TestWithSphere(SphereCollider* other);
	virtual CollisonTestResult TestWithCloth(ClothCollider* other);
};

