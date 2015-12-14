#pragma once

/*
	This component encapsulates whole Cloth Simulation funcionality.
	IMPORTANT: This component requires SimObject to have following properties:
		- First mesh of mesh collection exists and is MeshGLPlane
*/

#include "Component.h"
#include "Common.h"
#include "MeshGLPlane.h"
#include "Timer.h"
//#include <CL\opencl.h>

//#define VERTEX_NEIGHBOURING_VERTICES 12
#define ALMOST_ZERO 0.000000001f

/////////////////////////////////////////

struct SimData
{
	int m_vertexCount;
	int m_edgesWidthAll;
	int m_edgesLengthAll;

	glm::vec4* b_neighbours;
	glm::vec4* b_neighboursDiag;
	glm::vec4* b_neighbours2;
	glm::vec4* b_neighbourMultipliers;
	glm::vec4* b_neighbourDiagMultipliers;
	glm::vec4* b_neighbour2Multipliers;
	glm::vec4* b_positionLast;
	glm::vec4* b_elMassCoeffs;
	glm::vec4* b_multipliers;

	glm::vec4 c_springLengths;

	GLuint i_neighbours;
	GLuint i_neighboursDiag;
	GLuint i_neighbours2;
	GLuint i_neighbourMultipliers;
	GLuint i_neighbourDiagMultipliers;
	GLuint i_neighbour2Multipliers;
	GLuint i_position;
	GLuint i_positionLast;
	GLuint i_elMassCoeffs;
	GLuint i_multipliers;

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
		b_neighboursDiag = nullptr;
		b_neighbours2 = nullptr;
		b_neighbourMultipliers = nullptr;
		b_neighbourDiagMultipliers = nullptr;
		b_neighbour2Multipliers = nullptr;
		b_elMassCoeffs = nullptr;
		b_multipliers = nullptr;
		i_position = 0;
		i_positionLast = 0;
		i_elMassCoeffs = 0;
		i_multipliers = 0;
	}

	~SimData()
	{
		if(b_neighbours != nullptr)
			delete b_neighbours;
		if (b_neighboursDiag != nullptr)
			delete b_neighboursDiag;
		if (b_neighbours2 != nullptr)
			delete b_neighbours2;
		if (b_neighbourMultipliers != nullptr)
			delete b_neighbourMultipliers;
		if (b_neighbourDiagMultipliers != nullptr)
			delete b_neighbourDiagMultipliers;
		if (b_neighbour2Multipliers != nullptr)
			delete b_neighbour2Multipliers;
		if (b_elMassCoeffs != nullptr)
			delete b_elMassCoeffs;
		if (b_positionLast != nullptr)
			delete b_positionLast;
		if (b_multipliers != nullptr)
			delete b_multipliers;
	}
};

////////////////////////////////////

enum ClothSimulationMode
{
	MASS_SPRING,
	POSITION_BASED
};

////////////////////////////////////

class ClothSimulator :
	public Component
{
protected:
	const double FIXED_DELTA = 0.015f;
	const float VERTEX_MASS = 1.0f;
	const float VERTEX_AIR_DAMP = 0.01f;
	const float SPRING_ELASTICITY = 50.00f;
	const float SPRING_BORDER_MULTIPLIER = 20.0f;
	const float SPRING_ELASTICITY_DAMP = -100.25f;
	const float VERTEX_COLLIDER_MULTIPLIER = 0.5f;
	const float CELL_OFFSET = 0.01f;
	const float COLLISION_CHECK_WINDOW_SIZE = 2;

	const int MODE_COUNT = 2;

	const unsigned int KERNEL_NRM_OUTPUT_NAME_COUNT = 1;
	const std::string KERNEL_NRM_NAME = "ClothNormal";
	const char* KERNEL_NRM_INPUT_NAMES[1] =
	{
		"InPos"
	};
	const char* KERNEL_NRM_OUTPUT_NAMES[1] =
	{
		"OutNormal"
	};

	const unsigned int KERNEL_COL_OUTPUT_NAME_COUNT = 1;
	const std::string KERNEL_COL_NAME = "ClothCollision";
	const char* KERNEL_COL_INPUT_NAMES[3] =
	{
		"InPos",
		"InBaaCols",
		"InSCols"
	};
	const char* KERNEL_COL_OUTPUT_NAMES[1] =
	{
		"OutPos"
	};

	/////////

	const unsigned int KERNEL_MSPOS_OUTPUT_NAME_COUNT = 2;
	const char* KERNEL_MSPOS_INPUT_NAMES[2] =
	{
		"InPos",
		"InPosLast"
	};
	const char* KERNEL_MSPOS_OUTPUT_NAMES[2] =
	{
		"OutPos",
		"OutPosLast"
	};
	const std::string KERNEL_MSPOS_NAME = "ClothMSPosition";

	KernelID* m_msPosKernelID;
	GLuint m_msPtfID;

	/////////

	const unsigned int KERNEL_PBPOS_OUTPUT_NAME_COUNT = 2;
	const char* KERNEL_PBPOS_INPUT_NAMES[2] =
	{
		"InPos",
		"InPosLast"
	};
	const char* KERNEL_PBPOS_OUTPUT_NAMES[2] =
	{
		"OutPos",
		"OutPosLast"
	};
	const std::string KERNEL_PBPOS_NAME = "ClothPBPosition";

	KernelID* m_pbPosKernelID;
	GLuint m_pbPtfID;

	/////////

	ClothSimulationMode m_mode;

	MeshGLPlane* m_meshPlane;
	VertexData** m_vd;
	VertexData* m_vdCopy;
	SimData* m_simData;

	KernelID* m_normalsKernel;
	KernelID* m_collisionsKernel;

	int m_boxColliderCount;
	int m_sphereColliderCount;
	float m_cellSize;

	GLuint m_ntfID;
	GLuint m_ctfID;
	GLuint m_vaoUpdateID[2];
	GLuint m_vaoRenderID[2];
	GLuint m_vboPosID[2];
	GLuint m_vboPosLastID[2];
	GLuint m_vboNrmID[2];
	GLuint m_vboColBAAID;
	GLuint m_vboColSID;
	GLuint m_texPosID[2];
	GLuint m_texPosLastID[2];
	GLuint m_texNrmPosID[2];
	GLuint m_texColPosID[2];
	GLuint m_texColBAAID;
	GLuint m_texColSID;

	unsigned int m_writeID;
	unsigned int m_readID;

	bool m_ifRestart;

	double m_timeStartMS;
	double m_timeSimMS;

	virtual inline unsigned int InitializeSimMS();
	virtual inline unsigned int ShutdownSimMS();
	virtual inline unsigned int UpdateSimMS
		(
			float gravity, 
			float fixedDelta
		);

	virtual inline unsigned int InitializeSimPB();
	virtual inline unsigned int ShutdownSimPB();
	virtual inline unsigned int UpdateSimPB
		(
			float gravity,
			float fixedDelta
		);

	inline void CopyVertexData(VertexData* source, VertexData* dest);
	inline void SwapRWIds();
	inline void AppendRestart();
	inline void RestartSimulation();
public:
	ClothSimulator(SimObject* obj);
	ClothSimulator(const ClothSimulator* c);
	~ClothSimulator();

	virtual unsigned int Initialize();
	virtual unsigned int Shutdown();

	virtual unsigned int Update();
	virtual unsigned int Draw();

	void SetMode(ClothSimulationMode mode);
	void SwitchMode();

	ClothSimulationMode GetMode();
	double GetSimTimeMS();
};

