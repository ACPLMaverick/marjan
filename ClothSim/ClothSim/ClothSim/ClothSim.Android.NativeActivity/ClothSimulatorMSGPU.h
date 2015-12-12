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
	const char* KERNEL_SIM_INPUT_NAMES[2] =
	{
		"InPos",
		"InPosLast"
	};
	const char* KERNEL_SIM_OUTPUT_NAMES[2] =
	{
		"OutPos",
		"OutPosLast"
	};
	const std::string KERNEL_SIM_NAME = "ClothMSPosition";

	GLuint m_tfID;
	GLuint m_vaoUpdateID[2];
	GLuint m_vaoRenderID[2];
	GLuint m_vboPosID[2];
	GLuint m_vboPosLastID[2];
	GLuint m_texPosID[2];
	GLuint m_texPosLastID[2];

	unsigned int m_writeID;
	unsigned int m_readID;

	virtual inline unsigned int InitializeSim();
	virtual inline unsigned int ShutdownSim();
	virtual inline unsigned int UpdateSim(
		float gravity,
		float fixedDelta,
		BoxAAData* bColliders,
		SphereData* sColliders,
		glm::mat4* wm
		);

	inline void SwapRWIds();
public:
	ClothSimulatorMSGPU(SimObject* obj);
	ClothSimulatorMSGPU(const ClothSimulatorMSGPU* c);
	~ClothSimulatorMSGPU();
};

