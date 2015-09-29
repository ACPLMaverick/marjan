#include "SphereCollider.h"
#include "BoxAACollider.h"

SphereCollider::SphereCollider(SimObject* obj) : Collider(obj)
{
	m_offset = glm::vec3(0.0f, 0.0f, 0.0f);
	m_radius = 1.0f;
}

SphereCollider::SphereCollider(SimObject* obj, glm::vec3* offset, float radius) : Collider(obj)
{
	m_offset = *offset;
	m_radius = radius;
}

SphereCollider::~SphereCollider()
{
}



unsigned int SphereCollider::Initialize()
{
	unsigned int err = Collider::Initialize();
	if (err != CS_ERR_NONE)
		return err;

	return err;
}

unsigned int SphereCollider::Shutdown()
{
	unsigned int err = Collider::Shutdown();
	if (err != CS_ERR_NONE)
		return err;

	m_type = SPHERE;

	return err;
}

unsigned int SphereCollider::Update()
{
	return CS_ERR_NONE;
}

unsigned int SphereCollider::Draw()
{
	return CS_ERR_NONE;
}



CollisonTestResult SphereCollider::TestWithBoxAA(BoxAACollider* other)
{
	CollisonTestResult res;

	return res;
}

CollisonTestResult SphereCollider::TestWithSphere(SphereCollider* other)
{
	CollisonTestResult res;

	return res;
}