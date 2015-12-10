#include "pch.h"
#include "ClothSimulatorMSGPU.h"


ClothSimulatorMSGPU::ClothSimulatorMSGPU(SimObject * obj) : ClothSimulator(obj)
{
}

ClothSimulatorMSGPU::ClothSimulatorMSGPU(const ClothSimulatorMSGPU * c) : ClothSimulator(c)
{
}

ClothSimulatorMSGPU::~ClothSimulatorMSGPU()
{
}



inline unsigned int ClothSimulatorMSGPU::InitializeSim()
{
	unsigned int err = CS_ERR_NONE;

	// initialize kernels 
	m_kernelID = ResourceManager::GetInstance()->LoadKernel(&KERNEL_SIM_NAME);
	glGenTransformFeedbacks(1, &m_tfID);
	glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, m_tfID);
	glTransformFeedbackVaryings(m_kernelID->id, KERNEL_SIM_OUTPUT_NAME_COUNT, KERNEL_SIM_OUTPUT_NAMES, GL_SEPARATE_ATTRIBS);
	glLinkProgram(m_kernelID->id);

	// initialize method-related data

	return err;
}

inline unsigned int ClothSimulatorMSGPU::ShutdownSim()
{
	unsigned int err = CS_ERR_NONE;

	// shutdown kernels 
	
	// shutdown method-related data

	return err;
}

inline unsigned int ClothSimulatorMSGPU::UpdateSim(float gravity, float fixedDelta, BoxAAData * bColliders, SphereData * sColliders, glm::mat4 * wm)
{
	unsigned int err = CS_ERR_NONE;

	// call kernels

	return err;
}
