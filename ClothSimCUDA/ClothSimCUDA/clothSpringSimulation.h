#pragma once

#include "Common.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

/*
	Header file for Cloth Spring Simulation CUDA class.
*/

#define VERTEX_NEIGHBOURING_VERTICES 4

struct Vertex
{
	glm::vec3 force;
	glm::vec3 speed;

	unsigned int neighbours[VERTEX_NEIGHBOURING_VERTICES];

	unsigned int id;
	float mass;
	float lockMultiplier;

	glm::vec3* positionPtr;
	glm::vec2* uvPtr;
	glm::vec3* normalPtr;
	glm::vec4* colorPtr;

	Vertex()
	{
		force = glm::vec3(0.0f, 0.0f, 0.0f);
		speed = glm::vec3(0.0f, 0.0f, 0.0f);

		for (int i = 0; i < VERTEX_NEIGHBOURING_VERTICES; ++i)
		{
			neighbours[i] = 0xFFFFFFFF;
		}

		id = 0xFFFFFFFF;
		mass = 0.0f;
		lockMultiplier = 1.0f;

		positionPtr = nullptr;
		uvPtr = nullptr;
		normalPtr = nullptr;
		colorPtr = nullptr;
	}
};

struct Spring
{
	// positive accounts for first vertex, negative for second one
	glm::vec3 force;

	unsigned int idFirst;
	unsigned int idSecond;

	float baseLength;
	float elasticity;
};

class clothSpringSimulation
{
private:
	cudaDeviceProp* m_deviceProperties;

	const float VERTEX_MASS = 0.1f;
	const float SPRING_ELASTICITY = 5.0f;

	unsigned int m_vertexCount;
	unsigned int m_springCount;
	unsigned int m_vertexPositionSize;
	unsigned int m_vertexNormalSize;
	unsigned int m_vertexColorSize;
	unsigned int m_allEdgesWidth;
	unsigned int m_allEdgesLength;

	Vertex* m_vertices;
	Spring* m_springs;

	glm::vec3* m_posPtr;
	glm::vec3* m_nrmPtr;
	glm::vec4* m_colPtr;

	// device memory pointers
	Vertex* i_vertexPtr;
	Spring* i_springPtr;
	glm::vec3* i_posPtr;
	glm::vec3* i_nrmPtr;
	glm::vec4* i_colPtr;
	float* i_gravPtr;
	/////////////////

	inline cudaError_t CalculateForces(float gravity);
	inline void FreeMemory();
public:
	clothSpringSimulation();
	~clothSpringSimulation();

	unsigned int ClothSpringSimulationInitialize(
		unsigned int vertexPositionSize, 
		unsigned int vertexNormalSize, 
		unsigned int vertexColorSize,
		unsigned int edgesWidth,
		unsigned int edgesLength,
		glm::vec3* vertexPositionPtr, 
		glm::vec3* vertexNormalPtr, 
		glm::vec4* vertexColorPtr
		);
	unsigned int ClothSpringSimulationUpdate(float gravity);
	unsigned int ClothSpringSimulationShutdown();
};

