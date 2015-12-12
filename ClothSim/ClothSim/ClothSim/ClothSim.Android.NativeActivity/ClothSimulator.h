#pragma once

/*
	This component is an abstraction of whole Cloth Simulation funcionality.
	IMPORTANT: This component requires SimObject to have following properties:
		- First mesh of mesh collection exists and is MeshGLPlane
		- First collider of collider collection exists and is ClothCollider
*/

#include "Component.h"
#include "Common.h"
#include "MeshGLPlane.h"
#include "Timer.h"
//#include <CL\opencl.h>

#define VERTEX_NEIGHBOURING_VERTICES 4
#define COLLISION_CHECK_WINDOW_SIZE 4
#define ALMOST_ZERO 0.000000001f

/////////////////////////////////////////

struct SimData
{
	int m_vertexCount;
	int m_edgesWidthAll;
	int m_edgesLengthAll;

	float *b_neighbours;
	float *b_neighbourMultipliers;
	float *b_springLengths;

	glm::vec4* b_positionLast;
	float* b_elasticity;
	float* b_mass;
	float* b_dampCoeff;
	float* b_airDampCoeff;
	float* b_lockMultiplier;
	float* b_colliderMultiplier;

	GLuint i_neighbours;
	GLuint i_neighbourMultipliers;
	GLuint i_position;
	GLuint i_positionLast;
	GLuint i_springLengths;
	GLuint i_elasticity;
	GLuint i_mass;
	GLuint i_airDampCoeff;
	GLuint i_dampCoeff;
	GLuint i_lockMultiplier;
	GLuint i_colliderMultiplier;

	GLuint iu_vertexCount;
	GLuint iu_edgesWidthAll;
	GLuint iu_edgesLengthAll;
	GLuint iu_deltaTime;
	GLuint iu_gravity;

	SimData()
	{
		m_vertexCount = 0;
		m_edgesWidthAll = 0;
		m_edgesLengthAll = 0;
		b_neighbours = nullptr;
		b_neighbourMultipliers = nullptr;
		b_airDampCoeff = nullptr;
		b_positionLast = nullptr;
		b_springLengths = nullptr;
		b_elasticity = nullptr;
		b_mass = nullptr;
		b_dampCoeff = nullptr;
		b_lockMultiplier = nullptr;
		b_colliderMultiplier = nullptr;
		i_position = 0;
		i_positionLast = 0;
		i_springLengths = 0;
		i_elasticity = 0;
		i_mass = 0;
		i_dampCoeff = 0;
		i_lockMultiplier = 0;
		i_colliderMultiplier = 0;
	}

	~SimData()
	{
		if(b_neighbours != nullptr)
			delete b_neighbours;
		if (b_neighbourMultipliers != nullptr)
			delete b_neighbourMultipliers;
		if (b_airDampCoeff != nullptr)
			delete b_airDampCoeff;
		if (b_positionLast != nullptr)
			delete b_positionLast;
		if (b_springLengths != nullptr)
			delete b_springLengths;
		if (b_elasticity != nullptr)
			delete b_elasticity;
		if (b_mass != nullptr)
			delete b_mass;
		if (b_dampCoeff != nullptr)
			delete b_dampCoeff;
		if (b_lockMultiplier != nullptr)
			delete b_lockMultiplier;
		if (b_colliderMultiplier != nullptr)
			delete b_colliderMultiplier;
	}
};

////////////////////////////////////

class ClothSimulator :
	public Component
{
protected:
	const double FIXED_DELTA = 0.006f;
	const float VERTEX_MASS = 1.0f;
	const float VERTEX_AIR_DAMP = 0.01f;
	const float SPRING_ELASTICITY = 50.00f;
	const float SPRING_BORDER_MULTIPLIER = 50.0f;
	const float SPRING_ELASTICITY_DAMP = -100.25f;
	const float VERTEX_COLLIDER_MULTIPLIER = 0.5f;
	const float CELL_OFFSET = 0.01f;

	const unsigned int KERNEL_NRM_OUTPUT_NAME_COUNT = 1;
	const std::string KERNEL_NRM_NAME = "ClothNormals";
	const char* KERNEL_NRM_INPUT_NAMES[1] =
	{
		"InNormal"
	};
	const char* KERNEL_NRM_OUTPUT_NAMES[1] =
	{
		"OutNormal"
	};

	const unsigned int KERNEL_ECL_OUTPUT_NAME_COUNT = 1;
	const std::string KERNEL_ECL_NAME = "ClothExternalCollisions";
	const char* KERNEL_ECL_INPUT_NAMES[1] =
	{
		"InPos"
	};
	const char* KERNEL_ECL_OUTPUT_NAMES[1] =
	{
		"OutPos"
	};

	const unsigned int KERNEL_ICL_OUTPUT_NAME_COUNT = 1;
	const std::string KERNEL_ICL_NAME = "ClothInternalCollisions";
	const char* KERNEL_ICL_INPUT_NAMES[1] =
	{
		"InPos"
	};
	const char* KERNEL_ICL_OUTPUT_NAMES[1] =
	{
		"OutPos"
	};

	MeshGLPlane* m_meshPlane;
	VertexData** m_vd;
	VertexData* m_vdCopy;
	SimData* m_simData;

	KernelID* m_normalsKernel;
	KernelID* m_externalCollisionsKernel;
	KernelID* m_internalCollisonsKernel;

	int m_boxColliderCount;
	int m_sphereColliderCount;
	float m_cellSize;

	GLuint m_ntfID;
	GLuint m_eCtfID;
	GLuint m_iCtfID;
	GLuint m_vaoUpdateID[2];
	GLuint m_vaoRenderID[2];
	GLuint m_vboPosID[2];
	GLuint m_vboPosLastID[2];
	GLuint m_vboNrmID[2];
	GLuint m_texPosID[2];
	GLuint m_texPosLastID[2];
	GLuint m_texNrmPosID[2];
	GLuint m_texIcPosID[2];
	GLuint m_texEcPosID[2];

	unsigned int m_writeID;
	unsigned int m_readID;

	virtual inline unsigned int InitializeSim() = 0;
	virtual inline unsigned int ShutdownSim() = 0;
	virtual inline unsigned int UpdateSim
		(
		float gravity, 
		float fixedDelta
		) = 0;

	inline void CopyVertexData(VertexData* source, VertexData* dest);
	inline void SwapRWIds();
public:
	ClothSimulator(SimObject* obj);
	ClothSimulator(const ClothSimulator* c);
	~ClothSimulator();

	virtual unsigned int Initialize();
	virtual unsigned int Shutdown();

	virtual unsigned int Update();
	virtual unsigned int Draw();
};

