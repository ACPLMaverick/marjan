#pragma once
#include "ClothSimulator.h"

/*
This component encapsulates whole Cloth Simulation funcionality.
It uses mass-spring method to calculate cloth physics and calls necessary OpenGL kernels.
IMPORTANT: This component requires SimObject to have following properties:
- First mesh of mesh collection exists and is MeshGLPlane
- First collider of collider collection exists and is ClothCollider
*/

class ClothSimulatorMSGPU :
	public ClothSimulator
{
protected:
	const unsigned int KERNEL_SIM_OUTPUT_NAME_COUNT = 2;
	const char* KERNEL_SIM_OUTPUT_NAMES[2] =
	{
		"OutPos",
		"OutPosLast"
	};
	const std::string KERNEL_SIM_NAME = "ClothMSSimulation";

	virtual inline unsigned int InitializeSim();
	virtual inline unsigned int ShutdownSim();
	virtual inline unsigned int UpdateSim(
		float gravity,
		float fixedDelta,
		BoxAAData* bColliders,
		SphereData* sColliders,
		glm::mat4* wm
		);
public:
	ClothSimulatorMSGPU(SimObject* obj);
	ClothSimulatorMSGPU(const ClothSimulatorMSGPU* c);
	~ClothSimulatorMSGPU();
};

