#pragma once

/*
	This component encapsulates whole Cloth Simulation funcionality.
*/

#include "Component.h"
#include "Common.h"
#include "MeshGLPlane.h"
#include "Timer.h"
//#include <CL\opencl.h>

//#define VERTEX_NEIGHBOURING_VERTICES 12
#define ALMOST_ZERO 0.000000001f

////////////////////////////////////

enum ClothSimulationMode
{
	MASS_SPRING_GPU,
	MASS_SPRING_CPU,
	MASS_SPRING_CPUx4,
	POSITION_BASED_GPU,
	POSITION_BASED_CPU,
	POSITION_BASED_CPUx4,
	NONE,
};

/////////////////////////////////////////

struct SimParams
{
	ClothSimulationMode mode = ClothSimulationMode::NONE;
	glm::vec3 padding;
	float vertexMass = 1.0f;
	float vertexAirDamp = 0.01f;
	float elasticity = 50.00f;
	float elasticityDamp = -10.0f;
	float width = 10.0f;
	float length = 10.0f;
	unsigned int edgesWidth = 23;
	unsigned int edgesLength = 23;
};

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
	glm::vec4 c_touchVector;

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
		c_touchVector = glm::vec4(0.0f);
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

struct ThreadData
{
	ClothSimulator* inst;
	int diBegin;
	int diEnd;
};

////////////////////////////////////

class ClothSimulator :
	public Component
{
protected:
	SimParams m_simParams;
	SimData m_simData;

	const double FIXED_DELTA = 0.033f;
	const float MIN_MASS = 0.02f;
	const float SPRING_BORDER_MULTIPLIER = 20.0f;
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

	MeshGLPlane* m_meshPlane;
	VertexData** m_vd;
	VertexData** m_vdCopy;

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

	bool m_ifRestart = false;

	double m_timeStartMS;
	double m_timeSimMS;

	virtual inline unsigned int UpdateSimCPU
		(
			VertexData* vertexData,
			BoxAAData* boxAAData,
			SphereData* sphereData,
			int bcCount,
			int scCount,
			glm::mat4* worldMatrix,
			glm::mat4* viewMatrix,
			glm::mat4* projMatrix,
			float gravity,
			float fixedDelta
			);
	virtual inline unsigned int UpdateSimCPUx4
		(
			VertexData* vertexData,
			BoxAAData* boxAAData,
			SphereData* sphereData,
			int bcCount,
			int scCount,
			glm::mat4* worldMatrix,
			glm::mat4* viewMatrix,
			glm::mat4* projMatrix,
			float gravity,
			float fixedDelta
			);
	virtual inline unsigned int UpdateSimGPU
		(
			VertexData* vertexData,
			BoxAAData* boxAAData,
			SphereData* sphereData,
			int bcCount,
			int scCount,
			glm::mat4* worldMatrix,
			glm::mat4* viewMatrix,
			glm::mat4* projMatrix,
			float gravity,
			float fixedDelta
			);

	virtual inline unsigned int InitializeSimMSGPU();
	virtual inline unsigned int ShutdownSimMSGPU();
	virtual inline unsigned int UpdateSimMSGPU
		(
			float gravity, 
			float fixedDelta
		);

	virtual inline unsigned int InitializeSimPBGPU();
	virtual inline unsigned int ShutdownSimPBGPU();
	virtual inline unsigned int UpdateSimPBGPU
		(
			float gravity,
			float fixedDelta
		);

	virtual inline unsigned int UpdateSimMSCPU
		(
			float gravity,
			float fixedDelta
			);
	virtual inline unsigned int UpdateSimPBCPU
		(
			float gravity,
			float fixedDelta
			);

	inline void CalculateSpringForce
		(
			const glm::vec3 * mPos,
			const glm::vec3 * mPosLast,
			const glm::vec3 * nPos,
			const glm::vec3 * nPosLast,
			float sLength,
			float elCoeff,
			float dampCoeff,
			float fixedDelta,
			glm::vec3* ret
			);
	inline void CalcDistConstraint
		(
			glm::vec3* mPos,
			glm::vec3* nPos,
			float mass,
			float sLength,
			float elCoeff,
			glm::vec3* outConstraint,
			float* outW
			);

	inline void CopyVertexData(VertexData* source, VertexData* dest);
	inline void SwapRWIds();
	inline void AppendRestart();
	inline void RestartSimulation();

	static void UpdatePartial(void* args);
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
	void Restart();

	ClothSimulationMode GetMode();
	double GetSimTimeMS();

	void UpdateSimParams(SimParams* params);
	void UpdateTouchVector(const glm::vec2* pos, const glm::vec2* dir);
	SimParams* GetSimParams();
};

