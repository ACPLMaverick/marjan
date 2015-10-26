#pragma once

#include "Common.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

/*
	Header file for Cloth Spring Simulation CUDA class.
*/

#define VERTEX_NEIGHBOURING_VERTICES 4
#define ALMOST_ZERO 0.000000001f

struct Vertex
{
	glm::vec3 force;
	glm::vec3 velocity;
	glm::vec3 prevPosition;

	unsigned int neighbours[VERTEX_NEIGHBOURING_VERTICES];
	float neighbourMultipliers[VERTEX_NEIGHBOURING_VERTICES];
	glm::vec3 springLengths[VERTEX_NEIGHBOURING_VERTICES];

	unsigned int id;
	float mass;
	float elasticity;
	float dampCoeff;
	float lockMultiplier;

	Vertex()
	{
		force = glm::vec3(0.0f, 0.0f, 0.0f);
		velocity = glm::vec3(0.0f, 0.0f, 0.0f);
		prevPosition = glm::vec3(0.0f, 0.0f, 0.0f);

		for (int i = 0; i < VERTEX_NEIGHBOURING_VERTICES; ++i)
		{
			neighbours[i] = 0xFFFFFFFF;
			neighbourMultipliers[i] = 1.0f;
			springLengths[i] = glm::vec3(0.0f, 0.0f, 0.0f);
		}

		id = 0xFFFFFFFF;
		mass = 0.0f;
		dampCoeff = 0.0f;
		lockMultiplier = 1.0f;
		elasticity = 0.0f;
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

	const float VERTEX_MASS = 0.001f;
	const float VERTEX_DAMP = 0.01f;
	const float SPRING_ELASTICITY = 0.00001f;

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

	inline cudaError_t CalculateForces(float gravity, double delta, int steps);
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
	unsigned int ClothSpringSimulationUpdate(float gravity, double delta, int steps);
	unsigned int ClothSpringSimulationShutdown();
};

