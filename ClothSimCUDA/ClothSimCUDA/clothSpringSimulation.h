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

	float* i_posPtr;
	float* o_posPtr;
	float* i_nrmPtr;
	float* o_nrmPtr;
	float* i_colPtr;
	float* o_colPtr;
	float* i_gravPtr;

	// temp
	const int arraySize = 5;
	int a[5];
	int b[5];
	int c[5];
	int *dev_a = 0;
	int *dev_b = 0;
	int *dev_c = 0;
	////

	cudaError_t AddWithCuda(int *c, const int *a, const int *b, unsigned int size);
	inline void FreeMemory();
public:
	clothSpringSimulation();
	~clothSpringSimulation();

	unsigned int ClothSpringSimulationInitialize(unsigned int vertexCount, unsigned int vertexPositionSize, unsigned int vertexNormalSize, unsigned int vertexColorSize);
	unsigned int ClothSpringSimulationUpdate(glm::vec3* vertexPositionPtr, glm::vec3* vertexNormalPtr, glm::vec3* vertexColorPtr, float gravity);
	unsigned int ClothSpringSimulationShutdown();
};

