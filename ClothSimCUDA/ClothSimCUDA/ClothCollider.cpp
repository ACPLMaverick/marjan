#include "ClothCollider.h"


ClothCollider::ClothCollider(SimObject* obj) : Collider(obj)
{
}

ClothCollider::ClothCollider(const ClothCollider* other) : Collider(other)
{
}

ClothCollider::~ClothCollider()
{
}



unsigned int ClothCollider::Initialize()
{
	unsigned int err = Collider::Initialize();
	if (err != CS_ERR_NONE)
		return err;

	m_type = CLOTH;

	// acquiring mesh from our object

	if (
		m_obj->GetMesh(0) != nullptr
		)
	{
		m_meshPlane = (MeshGLPlane*)m_obj->GetMesh(0);
	}
	else return CS_ERR_CLOTHCOLLIDER_MESH_OBTAINING_ERROR;

	return err;
}

unsigned int ClothCollider::Shutdown()
{
	unsigned int err = Collider::Shutdown();
	if (err != CS_ERR_NONE)
		return err;

	return err;
}

unsigned int ClothCollider::Update()
{
	
	return CS_ERR_NONE;
}

unsigned int ClothCollider::Draw()
{
	return CS_ERR_NONE;
}



CollisonTestResult ClothCollider::TestWithBoxAA(BoxAACollider* other)
{
	CollisonTestResult res;

	return res;
}

CollisonTestResult ClothCollider::TestWithSphere(SphereCollider* other)
{
	CollisonTestResult res;

	return res;
}

CollisonTestResult ClothCollider::TestWithCloth(ClothCollider* other)
{
	CollisonTestResult res;

	return res;
}