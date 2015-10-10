#pragma once

/*
	This component encapsulates whole Cloth Simulation funcionality.
	It uses mass-spring method to calculate cloth physics and calls necessary CUDA kernels.
	IMPORTANT: This component requires SimObject to have following properties:
		- First mesh of mesh collection exists and is MeshGLPlane
		- First collider of collider collection exists and is ClothCollider
*/

#include "Component.h"
#include "Common.h"
#include "MeshGLPlane.h"
#include "ClothCollider.h"
#include "clothSpringSimulation.h"


class ClothSimulator :
	public Component
{
protected:
	clothSpringSimulation* m_simulator;

	MeshGLPlane* m_meshPlane;
	ClothCollider* m_collider;
public:
	ClothSimulator(SimObject* obj);
	ClothSimulator(const ClothSimulator* c);
	~ClothSimulator();

	virtual unsigned int Initialize();
	virtual unsigned int Shutdown();

	virtual unsigned int Update();
	virtual unsigned int Draw();
};

