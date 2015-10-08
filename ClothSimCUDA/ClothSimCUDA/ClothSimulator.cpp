#include "ClothSimulator.h"


ClothSimulator::ClothSimulator(SimObject* obj) : Component(obj)
{
}

ClothSimulator::ClothSimulator(const ClothSimulator* c) : Component(c)
{

}

ClothSimulator::~ClothSimulator()
{
}



unsigned int ClothSimulator::Initialize()
{
	unsigned int err = CS_ERR_NONE;

	// acquiring mesh and collider

	if (
		m_obj->GetMesh(0) != nullptr
		)
	{
		m_meshPlane = (MeshGLPlane*)m_obj->GetMesh(0);
	}
	else return CS_ERR_CLOTHSIMULATOR_MESH_OBTAINING_ERROR;

	if (
		m_obj->GetCollider(0) != nullptr
		)
	{
		m_collider = (ClothCollider*)m_obj->GetCollider(0);
	}
	else return CS_ERR_CLOTHSIMULATOR_COLLIDER_OBTAINING_ERROR;

	return err;
}

unsigned int ClothSimulator::Shutdown()
{
	unsigned int err = CS_ERR_NONE;

	return err;
}



unsigned int ClothSimulator::Update()
{
	unsigned int err = CS_ERR_NONE;

	FunctionsStart();

	return err;
}
unsigned int ClothSimulator::Draw()
{
	unsigned int err = CS_ERR_NONE;

	return err;
}