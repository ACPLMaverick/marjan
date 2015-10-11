#pragma once

#include "Common.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

/*
	Header file for Cloth Spring Simulation CUDA class.
*/



class clothSpringSimulation
{
private:
	unsigned int m_vertexCount;
	unsigned int m_vertexPositionSize;
	unsigned int m_vertexNormalSize;
	unsigned int m_vertexColorSize;
	unsigned int m_allEdgesWidth;
	unsigned int m_allEdgesLength;

	glm::vec3* i_posPtr;
	//float* o_posPtr;
	glm::vec3* i_nrmPtr;
	//float* o_nrmPtr;
	glm::vec4* i_colPtr;
	//float* o_colPtr;
	float* i_gravPtr;

	inline cudaError_t CalculateForces(glm::vec3* vertexPositionPtr, glm::vec3* vertexNormalPtr, glm::vec4* vertexColorPtr, float gravity);
	//inline cudaError_t AddWithCuda(int *c, const int *a, const int *b, unsigned int size);
	inline void FreeMemory();
public:
	clothSpringSimulation();
	~clothSpringSimulation();

	unsigned int ClothSpringSimulationInitialize(
		unsigned int vertexPositionSize, 
		unsigned int vertexNormalSize, 
		unsigned int vertexColorSize,
		unsigned int edgesWidth,
		unsigned int edgesLength
		);
	unsigned int ClothSpringSimulationUpdate(glm::vec3* vertexPositionPtr, glm::vec3* vertexNormalPtr, glm::vec4* vertexColorPtr, float gravity);
	unsigned int ClothSpringSimulationShutdown();
};

