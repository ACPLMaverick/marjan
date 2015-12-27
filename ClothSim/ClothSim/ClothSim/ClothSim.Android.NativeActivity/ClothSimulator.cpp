#include "ClothSimulator.h"

ClothSimulator::ClothSimulator(SimObject* obj) : Component(obj)
{
}

ClothSimulator::ClothSimulator(const ClothSimulator* c) : Component(c)
{

}

ClothSimulator::~ClothSimulator()
{
}



unsigned int ClothSimulator::Initialize()
{
	unsigned int err = CS_ERR_NONE;

	// acquiring mesh

	glm::vec4 tCol = glm::vec4(1.0f, 0.5f, 0.7f, 1.0f);
	m_meshPlane = new MeshGLPlane(m_obj, m_simParams.width, m_simParams.length, m_simParams.edgesWidth, m_simParams.edgesLength, &tCol);
	m_meshPlane->Initialize();
	m_meshPlane->SetGloss(10.0f);
	m_meshPlane->SetSpecular(0.2f);
	m_meshPlane->SetTextureID(ResourceManager::GetInstance()->GetTextureWhite());
	m_obj->AddMesh(m_meshPlane);

	// get a copy of start data
	m_vd = m_meshPlane->GetVertexDataDualPtr();
	m_vdCopy = new VertexData*[2];
	m_vdCopy[0] = new VertexData;
	m_vdCopy[1] = new VertexData;
	CopyVertexData(m_vd[0], m_vdCopy[0]);
	CopyVertexData(m_vd[1], m_vdCopy[1]);

	// initialize simulation data

	m_simData.m_vertexCount = m_vd[0]->data->vertexCount;
	m_simData.m_edgesWidthAll = m_meshPlane->GetEdgesWidth() + 2;
	m_simData.m_edgesLengthAll = m_meshPlane->GetEdgesLength() + 2;

	m_simData.b_positionLast = new glm::vec4[m_simData.m_vertexCount];
	m_simData.b_neighbours = new glm::vec4[m_simData.m_vertexCount];
	m_simData.b_neighboursDiag = new glm::vec4[m_simData.m_vertexCount];
	m_simData.b_neighbours2 = new glm::vec4[m_simData.m_vertexCount];
	m_simData.b_neighbourMultipliers = new glm::vec4[m_simData.m_vertexCount];
	m_simData.b_neighbourDiagMultipliers = new glm::vec4[m_simData.m_vertexCount];
	m_simData.b_neighbour2Multipliers = new glm::vec4[m_simData.m_vertexCount];
	m_simData.b_elMassCoeffs = new glm::vec4[m_simData.m_vertexCount];
	m_simData.b_multipliers = new glm::vec4[m_simData.m_vertexCount];

	glm::vec3 baseLength = glm::vec3(
		abs(m_vd[0]->data->positionBuffer[0].x - m_vd[0]->data->positionBuffer[m_simData.m_edgesWidthAll].x),
		0.0f,
		abs(m_vd[0]->data->positionBuffer[0].z - m_vd[0]->data->positionBuffer[1].z)
		);
	m_simData.c_springLengths.x = baseLength.x;	// two different spring lengths, horizontal and vertical
	m_simData.c_springLengths.y = baseLength.z;
	m_simData.c_springLengths.z = glm::sqrt(baseLength.x * baseLength.x + baseLength.z * baseLength.z);	// diagonal length
	m_simData.c_springLengths.w = 2.0f;	// multiplier for lenghts of neighbours2

	m_cellSize = glm::max(baseLength.x, baseLength.z) * 1.5f;

	for (int i = 0; i < m_simData.m_vertexCount; ++i)
	{
		///
		m_simData.b_neighboursDiag[i] = glm::vec4(0.0f);
		m_simData.b_neighbours2[i] = glm::vec4(0.0f);
		m_simData.b_neighbourDiagMultipliers[i] = glm::vec4(0.0f);
		m_simData.b_neighbour2Multipliers[i] = glm::vec4(0.0f);
		///
		m_simData.b_positionLast[i] = m_vd[0]->data->positionBuffer[i];
		m_simData.b_elMassCoeffs[i].y = m_simParams.vertexMass;
		m_simData.b_elMassCoeffs[i].w = m_simParams.vertexAirDamp;
		m_simData.b_multipliers[i].x = 1.0f;

		m_simData.b_multipliers[i].y = glm::min(glm::min(baseLength.x, baseLength.z) / 2.0f, VERTEX_COLLIDER_MULTIPLIER);

		m_simData.b_elMassCoeffs[i].x = m_simParams.elasticity;
		m_simData.b_elMassCoeffs[i].z = m_simParams.elasticityDamp;
		/*
		if (i < m_simData.m_edgesLengthAll ||
			i >= (m_simData.m_vertexCount - m_simData.m_edgesLengthAll) ||
			i % m_simData.m_edgesLengthAll == 0 ||
			i % m_simData.m_edgesLengthAll == (m_simData.m_edgesLengthAll - 1)
			)
		{
			m_simData.b_elMassCoeffs[i].x *= SPRING_BORDER_MULTIPLIER;
			m_simData.b_elMassCoeffs[i].z *= SPRING_BORDER_MULTIPLIER;
		}
		*/
		// calculating neighbouring vertices ids and spring lengths

		// upper
		m_simData.b_neighbours[i][0] = (i - 1) % m_simData.m_vertexCount;
		if (i % m_simData.m_edgesLengthAll)
		{
			m_simData.b_neighbourMultipliers[i][0] = 1.0f;
		}
		else
		{
			m_simData.b_neighbourMultipliers[i][0] = 0.0f;
		}

		// left
		m_simData.b_neighbours[i][1] = (i - m_simData.m_edgesLengthAll) % m_simData.m_vertexCount;
		if (i >= m_simData.m_edgesLengthAll)
		{
			m_simData.b_neighbourMultipliers[i][1] = 1.0f;
		}
		else
		{
			m_simData.b_neighbourMultipliers[i][1] = 0.0f;
		}

		// lower
		m_simData.b_neighbours[i][2] = (i + 1) % m_simData.m_vertexCount;
		if (i % m_simData.m_edgesLengthAll != (m_simData.m_edgesLengthAll - 1))
		{
			m_simData.b_neighbourMultipliers[i][2] = 1.0f;
		}
		else
		{
			m_simData.b_neighbourMultipliers[i][2] = 0.0f;
		}

		// right
		m_simData.b_neighbours[i][3] = (i + m_simData.m_edgesLengthAll) % m_simData.m_vertexCount;
		if (i < (m_simData.m_vertexCount - m_simData.m_edgesLengthAll))
		{
			m_simData.b_neighbourMultipliers[i][3] = 1.0f;
		}
		else
		{
			m_simData.b_neighbourMultipliers[i][3] = 0.0f;
		}

		// tl
		m_simData.b_neighboursDiag[i][0] = (i - m_simData.m_edgesLengthAll - 1) % m_simData.m_vertexCount;
		if (i >= m_simData.m_edgesLengthAll && i % m_simData.m_edgesLengthAll)
		{
			m_simData.b_neighbourDiagMultipliers[i][0] = 1.0f;
		}
		else
		{
			m_simData.b_neighbourDiagMultipliers[i][0] = 0.0f;
		}

		// bl
		m_simData.b_neighboursDiag[i][1] = (i - m_simData.m_edgesLengthAll + 1) % m_simData.m_vertexCount;
		if (i >= m_simData.m_edgesLengthAll && (i % m_simData.m_edgesLengthAll != (m_simData.m_edgesLengthAll - 1)))
		{
			m_simData.b_neighbourDiagMultipliers[i][1] = 1.0f;
		}
		else
		{
			m_simData.b_neighbourDiagMultipliers[i][1] = 0.0f;
		}

		// br
		m_simData.b_neighboursDiag[i][2] = (i + m_simData.m_edgesLengthAll + 1) % m_simData.m_vertexCount;
		if (i < (m_simData.m_vertexCount - m_simData.m_edgesLengthAll) && (i % m_simData.m_edgesLengthAll != (m_simData.m_edgesLengthAll - 1)))
		{
			m_simData.b_neighbourDiagMultipliers[i][2] = 1.0f;
		}
		else
		{
			m_simData.b_neighbourDiagMultipliers[i][2] = 0.0f;
		}

		// tr
		m_simData.b_neighboursDiag[i][3] = (i + m_simData.m_edgesLengthAll - 1) % m_simData.m_vertexCount;
		if (i < (m_simData.m_vertexCount - m_simData.m_edgesLengthAll) && i % m_simData.m_edgesLengthAll)
		{
			m_simData.b_neighbourDiagMultipliers[i][3] = 1.0f;
		}
		else
		{
			m_simData.b_neighbourDiagMultipliers[i][3] = 0.0f;
		}

		// upper2
		m_simData.b_neighbours2[i][0] = (i - 2) % m_simData.m_vertexCount;
		if (i % m_simData.m_edgesLengthAll > 1)
		{
			m_simData.b_neighbour2Multipliers[i][0] = 1.0f;
		}
		else
		{
			m_simData.b_neighbour2Multipliers[i][0] = 0.0f;
		}

		// left2
		m_simData.b_neighbours2[i][1] = (i - 2 * m_simData.m_edgesLengthAll) % m_simData.m_vertexCount;
		if (i >= 2 * m_simData.m_edgesLengthAll)
		{
			m_simData.b_neighbour2Multipliers[i][1] = 1.0f;
		}
		else
		{
			m_simData.b_neighbour2Multipliers[i][1] = 0.0f;
		}

		// lower2
		m_simData.b_neighbours2[i][2] = (i + 2) % m_simData.m_vertexCount;
		if (i % m_simData.m_edgesLengthAll < (m_simData.m_edgesLengthAll - 2))
		{
			m_simData.b_neighbour2Multipliers[i][2] = 1.0f;
		}
		else
		{
			m_simData.b_neighbour2Multipliers[i][2] = 0.0f;
		}

		// right2
		m_simData.b_neighbours2[i][3] = (i + 2 * m_simData.m_edgesLengthAll) % m_simData.m_vertexCount;
		if (i < (m_simData.m_vertexCount -  2 * m_simData.m_edgesLengthAll))
		{
			m_simData.b_neighbour2Multipliers[i][3] = 1.0f;
		}
		else
		{
			m_simData.b_neighbour2Multipliers[i][3] = 0.0f;
		}

		/*
		if (i == 0)
		{
			LOGI("%f %f %f %f", m_simData.b_neighbours[i][0], m_simData.b_neighbours[i][1], m_simData.b_neighbours[i][2], m_simData.b_neighbours[i][3]);
			LOGI("%f %f %f %f", m_simData.b_neighboursDiag[i][0], m_simData.b_neighboursDiag[i][1], m_simData.b_neighboursDiag[i][2], m_simData.b_neighboursDiag[i][3]);
			LOGI("%f %f %f %f", m_simData.b_neighbours2[i][0], m_simData.b_neighbours2[i][1], m_simData.b_neighbours2[i][2], m_simData.b_neighbours2[i][3]);
			LOGI("%f %f %f %f", m_simData.b_neighbourMultipliers[i][0], m_simData.b_neighbourMultipliers[i][1], m_simData.b_neighbourMultipliers[i][2], m_simData.b_neighbourMultipliers[i][3]);
			LOGI("%f %f %f %f", m_simData.b_neighbourDiagMultipliers[i][0], m_simData.b_neighbourDiagMultipliers[i][1], m_simData.b_neighbourDiagMultipliers[i][2], m_simData.b_neighbourDiagMultipliers[i][3]);
			LOGI("%f %f %f %f", m_simData.b_neighbour2Multipliers[i][0], m_simData.b_neighbour2Multipliers[i][1], m_simData.b_neighbour2Multipliers[i][2], m_simData.b_neighbour2Multipliers[i][3]);
		}
		*/
	}

	// hard-coded locks
	m_simData.b_multipliers[0][0] = 0.0f;
	m_simData.b_multipliers[(m_simData.m_vertexCount - m_simData.m_edgesLengthAll)][0] = 0.0f;

	m_readID = 0;
	m_writeID = 1;

	///////////////////////////////////////////////////////////////////////////////////////////////////

	// OpenGL Data initialization and binding
	m_vaoRenderID[0] = m_vd[0]->ids->vertexArrayID;
	m_vaoRenderID[1] = m_vd[1]->ids->vertexArrayID;
	glGenVertexArrays(2, m_vaoUpdateID);
	m_vboPosID[0] = m_vd[0]->ids->vertexBuffer;
	m_vboPosID[1] = m_vd[1]->ids->vertexBuffer;
	glGenBuffers(2, m_vboPosLastID);
	m_vboNrmID[0] = m_vd[0]->ids->normalBuffer;
	m_vboNrmID[1] = m_vd[1]->ids->normalBuffer;

	for (uint i = 0; i < 2; ++i)
	{
		glBindVertexArray(m_vaoUpdateID[i]);
		glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, m_vboPosID[i]);
		glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER,
			m_vd[0]->data->vertexCount * sizeof(m_vd[0]->data->positionBuffer[0]),
			&m_vd[0]->data->positionBuffer[0].x,
			GL_DYNAMIC_COPY);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, 0);

		glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, m_vboPosLastID[i]);
		glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER,
			m_simData.m_vertexCount * sizeof(m_simData.b_positionLast[0]),
			&m_simData.b_positionLast[0].x,
			GL_DYNAMIC_COPY);
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, 0);

		glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, m_vboNrmID[i]);
		glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER,
			m_vd[0]->data->vertexCount * sizeof(m_vd[0]->data->normalBuffer[0]),
			m_vd[0]->data->normalBuffer,
			GL_DYNAMIC_COPY);
		glEnableVertexAttribArray(2);
		glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 0, 0);
	}

	glGenBuffers(1, &m_simData.i_neighbours);
	glBindBuffer(GL_ARRAY_BUFFER, m_simData.i_neighbours);
	glBufferData(GL_ARRAY_BUFFER,
		m_simData.m_vertexCount * sizeof(m_simData.b_neighbours[0]),
		m_simData.b_neighbours,
		GL_STATIC_DRAW);
	glEnableVertexAttribArray(3);
	glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, 0, 0);

	glGenBuffers(1, &m_simData.i_neighboursDiag);
	glBindBuffer(GL_ARRAY_BUFFER, m_simData.i_neighboursDiag);
	glBufferData(GL_ARRAY_BUFFER,
		m_simData.m_vertexCount * sizeof(m_simData.b_neighboursDiag[0]),
		m_simData.b_neighboursDiag,
		GL_STATIC_DRAW);
	glEnableVertexAttribArray(4);
	glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, 0, 0);

	glGenBuffers(1, &m_simData.i_neighbours2);
	glBindBuffer(GL_ARRAY_BUFFER, m_simData.i_neighbours2);
	glBufferData(GL_ARRAY_BUFFER,
		m_simData.m_vertexCount * sizeof(m_simData.b_neighbours2[0]),
		m_simData.b_neighbours2,
		GL_STATIC_DRAW);
	glEnableVertexAttribArray(5);
	glVertexAttribPointer(5, 4, GL_FLOAT, GL_FALSE, 0, 0);

	glGenBuffers(1, &m_simData.i_neighbourMultipliers);
	glBindBuffer(GL_ARRAY_BUFFER, m_simData.i_neighbourMultipliers);
	glBufferData(GL_ARRAY_BUFFER,
		m_simData.m_vertexCount * sizeof(m_simData.b_neighbourMultipliers[0]),
		m_simData.b_neighbourMultipliers,
		GL_STATIC_DRAW);
	glEnableVertexAttribArray(6);
	glVertexAttribPointer(6, 4, GL_FLOAT, GL_FALSE, 0, 0);

	glGenBuffers(1, &m_simData.i_neighbourDiagMultipliers);
	glBindBuffer(GL_ARRAY_BUFFER, m_simData.i_neighbourDiagMultipliers);
	glBufferData(GL_ARRAY_BUFFER,
		m_simData.m_vertexCount * sizeof(m_simData.b_neighbourDiagMultipliers[0]),
		m_simData.b_neighbourDiagMultipliers,
		GL_STATIC_DRAW);
	glEnableVertexAttribArray(7);
	glVertexAttribPointer(7, 4, GL_FLOAT, GL_FALSE, 0, 0);

	glGenBuffers(1, &m_simData.i_neighbour2Multipliers);
	glBindBuffer(GL_ARRAY_BUFFER, m_simData.i_neighbour2Multipliers);
	glBufferData(GL_ARRAY_BUFFER,
		m_simData.m_vertexCount * sizeof(m_simData.b_neighbour2Multipliers[0]),
		m_simData.b_neighbour2Multipliers,
		GL_STATIC_DRAW);
	glEnableVertexAttribArray(8);
	glVertexAttribPointer(8, 4, GL_FLOAT, GL_FALSE, 0, 0);

	glGenBuffers(1, &m_simData.i_elMassCoeffs);
	glBindBuffer(GL_ARRAY_BUFFER, m_simData.i_elMassCoeffs);
	glBufferData(GL_ARRAY_BUFFER,
		m_simData.m_vertexCount * sizeof(m_simData.b_elMassCoeffs[0]),
		m_simData.b_elMassCoeffs,
		GL_STATIC_DRAW);
	glEnableVertexAttribArray(9);
	glVertexAttribPointer(9, 4, GL_FLOAT, GL_FALSE, 0, 0);

	glGenBuffers(1, &m_simData.i_multipliers);
	glBindBuffer(GL_ARRAY_BUFFER, m_simData.i_multipliers);
	glBufferData(GL_ARRAY_BUFFER,
		m_simData.m_vertexCount * sizeof(m_simData.b_multipliers[0]),
		m_simData.b_multipliers,
		GL_STATIC_DRAW);
	glEnableVertexAttribArray(10);
	glVertexAttribPointer(10, 4, GL_FLOAT, GL_FALSE, 0, 0);

	// Other kernel loading and feedbacks initialization goes here
	m_collisionsKernel = ResourceManager::GetInstance()->LoadKernel(&KERNEL_COL_NAME);

	glGenTransformFeedbacks(1, &m_ctfID);
	glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, m_ctfID);
	glTransformFeedbackVaryings(m_collisionsKernel->id, KERNEL_COL_OUTPUT_NAME_COUNT, KERNEL_COL_OUTPUT_NAMES, GL_SEPARATE_ATTRIBS);
	glLinkProgram(m_collisionsKernel->id);

	for (uint i = 0; i < 2; ++i)
	{
		m_texColPosID[i] = glGetUniformBlockIndex(m_collisionsKernel->id, KERNEL_COL_INPUT_NAMES[0]);
		glUniformBlockBinding(m_collisionsKernel->id, m_texColPosID[i], 1);
	}
	m_texColBAAID = glGetUniformBlockIndex(m_collisionsKernel->id, KERNEL_COL_INPUT_NAMES[1]);
	m_texColSID = glGetUniformBlockIndex(m_collisionsKernel->id, KERNEL_COL_INPUT_NAMES[2]);
	glUniformBlockBinding(m_collisionsKernel->id, m_texColBAAID, 0);
	glUniformBlockBinding(m_collisionsKernel->id, m_texColSID, 2);

	glGenBuffers(1, &m_vboColBAAID);
	glBindBuffer(GL_UNIFORM_BUFFER, m_vboColBAAID);
	glGenBuffers(1, &m_vboColSID);
	glBindBuffer(GL_UNIFORM_BUFFER, m_vboColSID);

	m_collisionsKernel->uniformIDs->push_back(glGetUniformLocation(m_collisionsKernel->id, "WorldMatrix"));
	m_collisionsKernel->uniformIDs->push_back(glGetUniformLocation(m_collisionsKernel->id, "GroundLevel"));
	m_collisionsKernel->uniformIDs->push_back(glGetUniformLocation(m_collisionsKernel->id, "BoxAAColliderCount"));
	m_collisionsKernel->uniformIDs->push_back(glGetUniformLocation(m_collisionsKernel->id, "SphereColliderCount"));
	m_collisionsKernel->uniformIDs->push_back(glGetUniformLocation(m_collisionsKernel->id, "EdgesWidthAll"));
	m_collisionsKernel->uniformIDs->push_back(glGetUniformLocation(m_collisionsKernel->id, "EdgesLengthAll"));
	m_collisionsKernel->uniformIDs->push_back(glGetUniformLocation(m_collisionsKernel->id, "VertexCount"));
	m_collisionsKernel->uniformIDs->push_back(glGetUniformLocation(m_collisionsKernel->id, "InternalCollisionCheckWindowSize"));

	////////////

	m_normalsKernel = ResourceManager::GetInstance()->LoadKernel(&KERNEL_NRM_NAME);

	glGenTransformFeedbacks(1, &m_ntfID);
	glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, m_ntfID);
	glTransformFeedbackVaryings(m_normalsKernel->id, KERNEL_NRM_OUTPUT_NAME_COUNT, KERNEL_NRM_OUTPUT_NAMES, GL_SEPARATE_ATTRIBS);
	glLinkProgram(m_normalsKernel->id);

	for (uint i = 0; i < 2; ++i)
	{
		m_texNrmPosID[i] = glGetUniformBlockIndex(m_normalsKernel->id, KERNEL_NRM_INPUT_NAMES[0]);
		glUniformBlockBinding(m_normalsKernel->id, m_texNrmPosID[i], 0);
	}

	// Sim-specific initialization. Sim-related transform feedback initialization and kernel loading goes here.
	err = InitializeSimMSGPU();
	err = InitializeSimPBGPU();

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_UNIFORM_BUFFER, 0);
	glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, 0);
	glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, 0);

	if (m_simParams.mode == ClothSimulationMode::MASS_SPRING_CPU || m_simParams.mode == ClothSimulationMode::POSITION_BASED_CPU)
	{
		m_meshPlane->SwapDataPtrs();
	}

	return err;
}

unsigned int ClothSimulator::Shutdown()
{
	unsigned int err = CS_ERR_NONE;

	m_meshPlane->Shutdown();
	m_obj->RemoveMesh(m_meshPlane);
	delete m_meshPlane;

	// Sim-specific shutdown
	err = ShutdownSimMSGPU();
	err = ShutdownSimPBGPU();

	glDeleteVertexArrays(2, m_vaoUpdateID);
	glDeleteBuffers(2, m_vboPosLastID);
	glDeleteBuffers(1, &m_vboColBAAID);
	glDeleteBuffers(1, &m_vboColSID);

	glDeleteBuffers(1, &m_simData.i_neighbours);
	glDeleteBuffers(1, &m_simData.i_neighboursDiag);
	glDeleteBuffers(1, &m_simData.i_neighbours2);
	glDeleteBuffers(1, &m_simData.i_neighbourMultipliers);
	glDeleteBuffers(1, &m_simData.i_neighbourDiagMultipliers);
	glDeleteBuffers(1, &m_simData.i_neighbour2Multipliers);
	glDeleteBuffers(1, &m_simData.i_elMassCoeffs);
	glDeleteBuffers(1, &m_simData.i_multipliers);

	glDeleteTransformFeedbacks(1, &m_ntfID);
	glDeleteTransformFeedbacks(1, &m_ctfID);

	delete m_vdCopy[0];
	delete m_vdCopy[1];
	delete[] m_vdCopy;


	return err;
}



unsigned int ClothSimulator::Update()
{
	unsigned int err = CS_ERR_NONE;

	if (m_ifRestart)
	{
		RestartSimulation();
		m_ifRestart = false;
	}

	/////////////////////////
	m_timeStartMS = Timer::GetInstance()->GetCurrentTimeMS();
	/////////////////////////

	VertexData* clothData = m_meshPlane->GetVertexDataPtr();
	BoxAAData* boxData = nullptr;
	SphereData* sphereData = nullptr;
	if (PhysicsManager::GetInstance()->GetBoxCollidersData()->size() != 0)
		boxData = &(PhysicsManager::GetInstance()->GetBoxCollidersData()->at(0));
	if (PhysicsManager::GetInstance()->GetSphereCollidersData()->size() != 0)
		sphereData = &(PhysicsManager::GetInstance()->GetSphereCollidersData()->at(0));
	int bcCount = PhysicsManager::GetInstance()->GetBoxCollidersData()->size();
	int scCount = PhysicsManager::GetInstance()->GetSphereCollidersData()->size();

	glm::mat4 zero = glm::mat4();
	glm::mat4* wm = &zero;

	if (m_obj->GetTransform() != nullptr)
		wm = m_obj->GetTransform()->GetWorldMatrix();

	if (m_simParams.mode == ClothSimulationMode::MASS_SPRING_CPU || m_simParams.mode == ClothSimulationMode::POSITION_BASED_CPU)
	{
		err = UpdateSimCPU(clothData, boxData, sphereData, bcCount, scCount, wm, PhysicsManager::GetInstance()->GetGravity(), (float)FIXED_DELTA);
	}
	else if (m_simParams.mode == ClothSimulationMode::MASS_SPRING_GPU || m_simParams.mode == ClothSimulationMode::POSITION_BASED_GPU)
	{
		err = UpdateSimGPU(clothData, boxData, sphereData, bcCount, scCount, wm, PhysicsManager::GetInstance()->GetGravity(), (float)FIXED_DELTA);
	}

	/////////////////////////
	double cTime = Timer::GetInstance()->GetCurrentTimeMS();
	m_timeSimMS = cTime - m_timeStartMS;
	/////////////////////////

	return err;
}
unsigned int ClothSimulator::Draw()
{
	unsigned int err = CS_ERR_NONE;

	return err;
}

void ClothSimulator::SetMode(ClothSimulationMode mode)
{
	if (mode != m_simParams.mode)
	{
		m_simParams.mode = mode;
		AppendRestart();
	}
}

void ClothSimulator::SwitchMode()
{
	m_simParams.mode = (ClothSimulationMode)(((int)m_simParams.mode + 1) % 2);
	AppendRestart();
}

void ClothSimulator::Restart()
{
	AppendRestart();
}

ClothSimulationMode ClothSimulator::GetMode()
{
	return m_simParams.mode;
}

double ClothSimulator::GetSimTimeMS()
{
	return m_timeSimMS;
}

void ClothSimulator::UpdateSimParams(SimParams * params)
{
	m_simParams = *params;
}

SimParams * ClothSimulator::GetSimParams()
{
	return &m_simParams;
}

inline unsigned int ClothSimulator::UpdateSimCPU
	(
		VertexData* vertexData,
		BoxAAData* boxAAData,
		SphereData* sphereData,
		int bcCount,
		int scCount,
		glm::mat4* worldMatrix,
		float gravity,
		float fixedDelta
	)
{
	unsigned int err = CS_ERR_NONE;

	// execute simulation procedure
	if (m_simParams.mode == ClothSimulationMode::MASS_SPRING_CPU)
	{
		err = UpdateSimMSCPU(
			PhysicsManager::GetInstance()->GetGravity(),
			(float)FIXED_DELTA
			);
	}
	else if (m_simParams.mode == ClothSimulationMode::POSITION_BASED_CPU)
	{
		err = UpdateSimPBCPU(
			PhysicsManager::GetInstance()->GetGravity(),
			(float)FIXED_DELTA
			);
	}

	SwapRWIds();

	// compute collisions and finger movement
	for (int i = 0; i < m_simData.m_vertexCount; ++i)
	{
		glm::vec3 colOffset = glm::vec3(0.0f);
		glm::vec4 modelPos = m_vd[m_readID]->data->positionBuffer[i];
		glm::vec3 worldPos = glm::vec3(*worldMatrix * modelPos);
		glm::vec3 screenPos = glm::vec3(0.0f);	// !
		glm::vec3 mPos = glm::vec3(modelPos);
		glm::vec4 lPos = m_vdCopy[m_readID]->data->positionBuffer[i];
		glm::vec3 totalOffset = glm::vec3(0.0f);
		float mR = m_simData.b_multipliers[i].y;

		// external collisons
		for (int j = 0; j < bcCount; ++j)
		{
			colOffset = glm::vec3(0.0f);
			BoxAAData col = boxAAData[j];
			glm::vec3 closest = glm::min(glm::max(worldPos, col.min), col.max);
			glm::vec3 diff = closest - worldPos;
			float dist = Vec3LengthSquared(&diff);

			if (dist < (mR * mR) && dist != 0.0f)
			{
				closest = worldPos - closest;
				colOffset = normalize(closest) * (mR - sqrt(dist));
				worldPos += colOffset;
				totalOffset += colOffset;
			}
		}

		for (int j = 0; j < scCount; ++j)
		{
			colOffset = glm::vec3(0.0f);
			SphereData col = sphereData[j];
			glm::vec3 diff = worldPos - col.center;
			float diffLength = Vec3LengthSquared(&diff);
			if
				(
					diffLength < (mR + col.radius) * (mR + col.radius) &&
					diffLength != 0.0f
					)
			{
				diff = normalize(diff);
				diff = diff * ((mR + col.radius) - sqrt(diffLength));

				colOffset = diff;
				worldPos += colOffset;
				totalOffset += colOffset;
			}
		}

		// internal collisions
		for (int j = 0; j < 4; ++j)
		{
			colOffset = glm::vec3(0.0f);
			glm::vec3 nPos = glm::vec3(*worldMatrix * m_vd[m_readID]->data->positionBuffer[(int)(glm::roundEven(m_simData.b_neighbours[i][j]))]);
			glm::vec3 diff = worldPos - nPos;
			float diffLength = Vec3LengthSquared(&diff);
			if
				(
					diffLength < (mR + mR) * (mR + mR) &&
					diffLength != 0.0f
					)
			{
				diff = normalize(diff);
				diff = diff * ((mR + mR) - sqrt(diffLength)) * 0.5f;

				colOffset = diff;
				worldPos += colOffset;
				totalOffset += colOffset;
			}
		}

		totalOffset *= m_simData.b_multipliers[i].x;
		// finger movement
		// TBA

		m_vd[m_writeID]->data->positionBuffer[i] = glm::vec4(mPos + totalOffset, 1.0f);
		m_vdCopy[m_writeID]->data->positionBuffer[i] = lPos;
	}
	SwapRWIds();

	// compute normals
	for (int i = 0; i < m_simData.m_vertexCount; ++i)
	{
		glm::vec3 normal = glm::vec3(0.0f);
		glm::vec3 mPos = glm::vec3(m_vd[m_readID]->data->positionBuffer[i]);

		for (int j = 0; j < 4; ++j)
		{
			int nID1 = (int)(glm::roundEven(m_simData.b_neighbours[i][j]));
			int nID2 = (int)(glm::roundEven(m_simData.b_neighbours[i][(j + 1) % 4]));
			glm::vec3 diff1 = mPos - glm::vec3(m_vd[m_readID]->data->positionBuffer[nID1]);
			glm::vec3 diff2 = mPos - glm::vec3(m_vd[m_readID]->data->positionBuffer[nID2]);

			glm::vec3 pNormal = (glm::cross(diff1, diff2) * m_simData.b_neighbourMultipliers[i][j] *
				m_simData.b_neighbourMultipliers[i][(j + 1) % 4]);
			if (pNormal.x != pNormal.x || pNormal.y != pNormal.y || pNormal.z != pNormal.z)
				pNormal = glm::vec3(0.0f, 1.0f, 0.0f);
			normal += pNormal;
		}
		normal = glm::normalize(normal);
		normal.z = -normal.z;
		m_vd[m_writeID]->data->normalBuffer[i] = glm::vec4(normal, 1.0f);
	}

	// update GPU buffers
	glBindVertexArray(m_vd[m_writeID]->ids->vertexArrayID);
	glBindBuffer(GL_ARRAY_BUFFER, m_vd[m_writeID]->ids->vertexBuffer);
	glBufferSubData(GL_ARRAY_BUFFER, 0,
		m_simData.m_vertexCount * sizeof(m_vd[m_writeID]->data->positionBuffer[0]),
		m_vd[m_writeID]->data->positionBuffer);

	glBindBuffer(GL_ARRAY_BUFFER, m_vd[m_writeID]->ids->normalBuffer);
	glBufferSubData(GL_ARRAY_BUFFER, 0,
		m_simData.m_vertexCount * sizeof(m_vd[m_writeID]->data->normalBuffer[0]),
		m_vd[m_writeID]->data->normalBuffer);

	//SwapRWIds();
	//m_meshPlane->SwapDataPtrs();

	return err;
}

inline unsigned int ClothSimulator::UpdateSimGPU
	(
		VertexData* vertexData,
		BoxAAData* boxAAData,
		SphereData* sphereData,
		int bcCount,
		int scCount,
		glm::mat4* worldMatrix,
		float gravity,
		float fixedDelta
	)
{
	unsigned int err = CS_ERR_NONE;
	ShaderID* cShaderID = Renderer::GetInstance()->GetCurrentShaderID();


	glBindVertexArray(m_vaoUpdateID[m_readID]);
	glBindBuffer(GL_ARRAY_BUFFER, m_vboPosID[m_readID]);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, 0);
	glBindBuffer(GL_ARRAY_BUFFER, m_vboPosLastID[m_readID]);
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, 0);
	glBindBuffer(GL_ARRAY_BUFFER, m_vd[m_readID]->ids->normalBuffer);
	glEnableVertexAttribArray(2);
	glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 0, 0);
	glBindBuffer(GL_ARRAY_BUFFER, m_simData.i_neighbours);
	glEnableVertexAttribArray(3);
	glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, 0, 0);
	glBindBuffer(GL_ARRAY_BUFFER, m_simData.i_neighboursDiag);
	glEnableVertexAttribArray(4);
	glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, 0, 0);
	glBindBuffer(GL_ARRAY_BUFFER, m_simData.i_neighbours2);
	glEnableVertexAttribArray(5);
	glVertexAttribPointer(5, 4, GL_FLOAT, GL_FALSE, 0, 0);
	glBindBuffer(GL_ARRAY_BUFFER, m_simData.i_neighbourMultipliers);
	glEnableVertexAttribArray(6);
	glVertexAttribPointer(6, 4, GL_FLOAT, GL_FALSE, 0, 0);
	glBindBuffer(GL_ARRAY_BUFFER, m_simData.i_neighbourDiagMultipliers);
	glEnableVertexAttribArray(7);
	glVertexAttribPointer(7, 4, GL_FLOAT, GL_FALSE, 0, 0);
	glBindBuffer(GL_ARRAY_BUFFER, m_simData.i_neighbour2Multipliers);
	glEnableVertexAttribArray(8);
	glVertexAttribPointer(8, 4, GL_FLOAT, GL_FALSE, 0, 0);
	glBindBuffer(GL_ARRAY_BUFFER, m_simData.i_elMassCoeffs);
	glEnableVertexAttribArray(9);
	glVertexAttribPointer(9, 4, GL_FLOAT, GL_FALSE, 0, 0);
	glBindBuffer(GL_ARRAY_BUFFER, m_simData.i_multipliers);
	glEnableVertexAttribArray(10);
	glVertexAttribPointer(10, 4, GL_FLOAT, GL_FALSE, 0, 0);

	// execute simulation kernel
	if (m_simParams.mode == ClothSimulationMode::MASS_SPRING_GPU)
	{
		err = UpdateSimMSGPU(
			PhysicsManager::GetInstance()->GetGravity(),
			(float)FIXED_DELTA
			);
	}
	else if (m_simParams.mode == ClothSimulationMode::POSITION_BASED_GPU)
	{
		err = UpdateSimPBGPU(
			PhysicsManager::GetInstance()->GetGravity(),
			(float)FIXED_DELTA
			);
	}

	SwapRWIds();

	// execute collision computation kernel

	glUseProgram(m_collisionsKernel->id);

	glDisableVertexAttribArray(0);
	glBindBufferBase(GL_UNIFORM_BUFFER, m_texColPosID[m_readID], 0);
	glBindBuffer(GL_ARRAY_BUFFER, m_vboPosID[m_readID]);
	glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, 0);
	glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 1, 0);

	glUniformMatrix4fv(m_collisionsKernel->uniformIDs->at(0), 1, GL_FALSE, (float*)worldMatrix);
	glUniform1f(m_collisionsKernel->uniformIDs->at(1), System::GetInstance()->GetCurrentScene()->GetGroundLevel());
	glUniform1i(m_collisionsKernel->uniformIDs->at(2), bcCount);
	glUniform1i(m_collisionsKernel->uniformIDs->at(3), scCount);
	glUniform1i(m_collisionsKernel->uniformIDs->at(4), m_simData.m_edgesWidthAll);
	glUniform1i(m_collisionsKernel->uniformIDs->at(5), m_simData.m_edgesLengthAll);
	glUniform1i(m_collisionsKernel->uniformIDs->at(6), m_simData.m_vertexCount);
	glUniform1i(m_collisionsKernel->uniformIDs->at(7), COLLISION_CHECK_WINDOW_SIZE);

	glBindBufferBase(GL_UNIFORM_BUFFER, m_texColPosID[m_readID], m_vboPosID[m_readID]);

	glBindBuffer(GL_UNIFORM_BUFFER, m_vboColBAAID);
	glBufferData(GL_UNIFORM_BUFFER, bcCount * sizeof(BoxAAData), boxAAData, GL_DYNAMIC_COPY);
	glBindBufferBase(GL_UNIFORM_BUFFER, m_texColBAAID, m_vboColBAAID);

	glBindBuffer(GL_UNIFORM_BUFFER, m_vboColSID);
	glBufferData(GL_UNIFORM_BUFFER, scCount * sizeof(SphereData), sphereData, GL_DYNAMIC_COPY);
	glBindBufferBase(GL_UNIFORM_BUFFER, m_texColSID, m_vboColSID);


	glBindBuffer(GL_ARRAY_BUFFER, m_vboPosID[m_readID]);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, 0);

	glEnable(GL_RASTERIZER_DISCARD);

	glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, m_ctfID);

	glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, m_vboPosID[m_readID]);

	glBeginTransformFeedback(GL_POINTS);
	glDrawArrays(GL_POINTS, 0, m_simData.m_vertexCount);
	glEndTransformFeedback();

	glDisable(GL_RASTERIZER_DISCARD);

	glBindBufferBase(GL_UNIFORM_BUFFER, m_texColPosID[m_readID], 0);
	glBindBuffer(GL_ARRAY_BUFFER, m_vboPosID[m_readID]);
	glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, 0);

	SwapRWIds();

	// execute normal computation kernel

	glUseProgram(m_normalsKernel->id);

	glDisableVertexAttribArray(0);
	glBindBufferBase(GL_UNIFORM_BUFFER, m_texNrmPosID[m_readID], 0);
	glBindBuffer(GL_ARRAY_BUFFER, m_vboPosID[m_readID]);
	glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, 0);
	glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 1, 0);

	glBindBufferBase(GL_UNIFORM_BUFFER, m_texNrmPosID[m_readID], m_vboPosID[m_readID]);

	glBindBuffer(GL_ARRAY_BUFFER, m_vboPosID[m_readID]);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, 0);

	glEnable(GL_RASTERIZER_DISCARD);

	glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, m_ntfID);

	glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, m_vboNrmID[m_writeID]);

	glBeginTransformFeedback(GL_POINTS);
	glDrawArrays(GL_POINTS, 0, m_simData.m_vertexCount);
	glEndTransformFeedback();

	glDisable(GL_RASTERIZER_DISCARD);

	SwapRWIds();

	glBindBufferBase(GL_UNIFORM_BUFFER, m_texNrmPosID[m_readID], 0);
	glBindBuffer(GL_ARRAY_BUFFER, m_vboPosID[m_readID]);
	glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, 0);
	glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, 0);

	/////////////////////

	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);
	glDisableVertexAttribArray(2);
	glDisableVertexAttribArray(3);
	glDisableVertexAttribArray(4);
	glDisableVertexAttribArray(5);
	glDisableVertexAttribArray(6);
	glDisableVertexAttribArray(7);

	//SwapRWIds();

	m_meshPlane->SwapDataPtrs();

	glUseProgram(cShaderID->id);

	return err;
}

////////////////////// MASS SPRING 

inline unsigned int ClothSimulator::InitializeSimMSGPU()
{
	unsigned int err = CS_ERR_NONE;

	m_msPosKernelID = ResourceManager::GetInstance()->LoadKernel(&KERNEL_MSPOS_NAME);
	glGenTransformFeedbacks(1, &m_msPtfID);
	glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, m_msPtfID);
	glTransformFeedbackVaryings(m_msPosKernelID->id, KERNEL_MSPOS_OUTPUT_NAME_COUNT, KERNEL_MSPOS_OUTPUT_NAMES, GL_SEPARATE_ATTRIBS);
	glLinkProgram(m_msPosKernelID->id);

	for (uint i = 0; i < 2; ++i)
	{
		m_texPosID[i] = glGetUniformBlockIndex(m_msPosKernelID->id, KERNEL_MSPOS_INPUT_NAMES[0]);
		m_texPosLastID[i] = glGetUniformBlockIndex(m_msPosKernelID->id, KERNEL_MSPOS_INPUT_NAMES[1]);
		glUniformBlockBinding(m_msPosKernelID->id, m_texPosID[i], 1);
		glUniformBlockBinding(m_msPosKernelID->id, m_texPosID[i], 0);
	}

	m_msPosKernelID->uniformIDs->push_back(glGetUniformLocation(m_msPosKernelID->id, "VertexCount"));
	m_msPosKernelID->uniformIDs->push_back(glGetUniformLocation(m_msPosKernelID->id, "EdgesWidthAll"));
	m_msPosKernelID->uniformIDs->push_back(glGetUniformLocation(m_msPosKernelID->id, "EdgesLengthAll"));
	m_msPosKernelID->uniformIDs->push_back(glGetUniformLocation(m_msPosKernelID->id, "DeltaTime"));
	m_msPosKernelID->uniformIDs->push_back(glGetUniformLocation(m_msPosKernelID->id, "Gravity"));
	m_msPosKernelID->uniformIDs->push_back(glGetUniformLocation(m_msPosKernelID->id, "SpringLengths"));

	return err;
}

inline unsigned int ClothSimulator::ShutdownSimMSGPU()
{
	unsigned int err = CS_ERR_NONE;

	// shutdown kernels 
	glDeleteTransformFeedbacks(1, &m_msPtfID);

	// shutdown method-related data

	return err;
}

inline unsigned int ClothSimulator::UpdateSimMSGPU(float gravity, float fixedDelta)
{
	unsigned int err = CS_ERR_NONE;

	glUseProgram(m_msPosKernelID->id);

	glUniform1i(m_msPosKernelID->uniformIDs->at(0), m_simData.m_vertexCount);
	glUniform1i(m_msPosKernelID->uniformIDs->at(1), m_simData.m_edgesWidthAll);
	glUniform1i(m_msPosKernelID->uniformIDs->at(2), m_simData.m_edgesLengthAll);
	glUniform1f(m_msPosKernelID->uniformIDs->at(3), fixedDelta);
	glUniform1f(m_msPosKernelID->uniformIDs->at(4), gravity);
	glUniform4f(m_msPosKernelID->uniformIDs->at(5), 
		m_simData.c_springLengths.x, 
		m_simData.c_springLengths.y, 
		m_simData.c_springLengths.z,
		m_simData.c_springLengths.w);

	glBindBufferBase(GL_UNIFORM_BUFFER, m_texPosID[m_readID], m_vboPosID[m_readID]);
	glBindBufferBase(GL_UNIFORM_BUFFER, m_texPosLastID[m_readID], m_vboPosLastID[m_readID]);

	glEnable(GL_RASTERIZER_DISCARD);

	glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, m_msPtfID);

	glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, m_vboPosID[m_writeID]);
	glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 1, m_vboPosLastID[m_writeID]);

	glBeginTransformFeedback(GL_POINTS);
	glDrawArrays(GL_POINTS, 0, m_simData.m_vertexCount);
	glEndTransformFeedback();

	glDisable(GL_RASTERIZER_DISCARD);
	/*
	glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, m_vboPosID[m_writeID]);
	glm::vec4* ptr = (glm::vec4*)glMapBufferRange(GL_TRANSFORM_FEEDBACK_BUFFER, 0, 1, GL_MAP_READ_BIT);
	for (int i = 0; i < 1; ++i, ++ptr)
	LOGW("%f %f %f %f", ptr->x, ptr->y, ptr->z, ptr->w);
	glUnmapBuffer(GL_TRANSFORM_FEEDBACK_BUFFER);
	*/

	glBindBufferBase(GL_UNIFORM_BUFFER, m_texPosID[m_readID], 0);
	glBindBufferBase(GL_UNIFORM_BUFFER, m_texPosLastID[m_readID], 0);

	return err;
}

//////////////////////////////

//////////////////// POSITION BASED

inline unsigned int ClothSimulator::InitializeSimPBGPU()
{
	unsigned int err = CS_ERR_NONE;

	m_pbPosKernelID = ResourceManager::GetInstance()->LoadKernel(&KERNEL_PBPOS_NAME);
	glGenTransformFeedbacks(1, &m_pbPtfID);
	glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, m_pbPtfID);
	glTransformFeedbackVaryings(m_pbPosKernelID->id, KERNEL_PBPOS_OUTPUT_NAME_COUNT, KERNEL_PBPOS_OUTPUT_NAMES, GL_SEPARATE_ATTRIBS);
	glLinkProgram(m_pbPosKernelID->id);

	for (uint i = 0; i < 2; ++i)
	{
		m_texPosID[i] = glGetUniformBlockIndex(m_pbPosKernelID->id, KERNEL_PBPOS_INPUT_NAMES[0]);
		m_texPosLastID[i] = glGetUniformBlockIndex(m_pbPosKernelID->id, KERNEL_PBPOS_INPUT_NAMES[1]);
		glUniformBlockBinding(m_pbPosKernelID->id, m_texPosID[i], 1);
		glUniformBlockBinding(m_pbPosKernelID->id, m_texPosID[i], 0);
	}

	m_pbPosKernelID->uniformIDs->push_back(glGetUniformLocation(m_pbPosKernelID->id, "VertexCount"));
	m_pbPosKernelID->uniformIDs->push_back(glGetUniformLocation(m_pbPosKernelID->id, "EdgesWidthAll"));
	m_pbPosKernelID->uniformIDs->push_back(glGetUniformLocation(m_pbPosKernelID->id, "EdgesLengthAll"));
	m_pbPosKernelID->uniformIDs->push_back(glGetUniformLocation(m_pbPosKernelID->id, "DeltaTime"));
	m_pbPosKernelID->uniformIDs->push_back(glGetUniformLocation(m_pbPosKernelID->id, "Gravity"));
	m_pbPosKernelID->uniformIDs->push_back(glGetUniformLocation(m_pbPosKernelID->id, "SpringLengths"));

	return err;
}

inline unsigned int ClothSimulator::ShutdownSimPBGPU()
{
	unsigned int err = CS_ERR_NONE;

	// shutdown kernels 
	glDeleteTransformFeedbacks(1, &m_pbPtfID);

	// shutdown method-related data

	return err;
}

inline unsigned int ClothSimulator::UpdateSimPBGPU(float gravity, float fixedDelta)
{
	unsigned int err = CS_ERR_NONE;

	glUseProgram(m_pbPosKernelID->id);

	glUniform1i(m_pbPosKernelID->uniformIDs->at(0), m_simData.m_vertexCount);
	glUniform1i(m_pbPosKernelID->uniformIDs->at(1), m_simData.m_edgesWidthAll);
	glUniform1i(m_pbPosKernelID->uniformIDs->at(2), m_simData.m_edgesLengthAll);
	glUniform1f(m_pbPosKernelID->uniformIDs->at(3), fixedDelta);
	glUniform1f(m_pbPosKernelID->uniformIDs->at(4), gravity);
	glUniform4f(m_pbPosKernelID->uniformIDs->at(5),
		m_simData.c_springLengths.x,
		m_simData.c_springLengths.y,
		m_simData.c_springLengths.z,
		m_simData.c_springLengths.w);

	glBindBufferBase(GL_UNIFORM_BUFFER, m_texPosID[m_readID], m_vboPosID[m_readID]);
	glBindBufferBase(GL_UNIFORM_BUFFER, m_texPosLastID[m_readID], m_vboPosLastID[m_readID]);

	glEnable(GL_RASTERIZER_DISCARD);

	glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, m_pbPtfID);

	glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, m_vboPosID[m_writeID]);
	glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 1, m_vboPosLastID[m_writeID]);

	glBeginTransformFeedback(GL_POINTS);
	glDrawArrays(GL_POINTS, 0, m_simData.m_vertexCount);
	glEndTransformFeedback();

	glDisable(GL_RASTERIZER_DISCARD);
	/*
	glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, m_vboPosID[m_writeID]);
	glm::vec4* ptr = (glm::vec4*)glMapBufferRange(GL_TRANSFORM_FEEDBACK_BUFFER, 0, 1, GL_MAP_READ_BIT);
	for (int i = 0; i < 1; ++i, ++ptr)
	LOGW("%f %f %f %f", ptr->x, ptr->y, ptr->z, ptr->w);
	glUnmapBuffer(GL_TRANSFORM_FEEDBACK_BUFFER);
	*/

	glBindBufferBase(GL_UNIFORM_BUFFER, m_texPosID[m_readID], 0);
	glBindBufferBase(GL_UNIFORM_BUFFER, m_texPosLastID[m_readID], 0);

	return err;
}

inline unsigned int ClothSimulator::UpdateSimMSCPU(float gravity, float fixedDelta)
{
	unsigned int err = CS_ERR_NONE;

	glm::vec3 posCurrent, posLast, force, tempForce, velocity, posNew, nPos, nPosLast, nVelocity;
	float elasticity, elDamp, airDamp, mass, lockMultiplier;
	glm::vec4 sls1 = 
	{
		m_simData.c_springLengths.y, 
		m_simData.c_springLengths.x, 
		m_simData.c_springLengths.y, 
		m_simData.c_springLengths.x
	};
	glm::vec4 sls2 =
	{
		m_simData.c_springLengths.z,
		m_simData.c_springLengths.z,
		m_simData.c_springLengths.z,
		m_simData.c_springLengths.z
	};
	glm::vec4 sls3 =
	{
		m_simData.c_springLengths.y * m_simData.c_springLengths.w,
		m_simData.c_springLengths.x * m_simData.c_springLengths.w,
		m_simData.c_springLengths.y * m_simData.c_springLengths.w,
		m_simData.c_springLengths.x * m_simData.c_springLengths.w
	};

	for (int i = 0; i < m_simData.m_vertexCount; ++i)
	{
		posCurrent = glm::vec3(m_vd[m_readID]->data->positionBuffer[i]);
		posLast = glm::vec3(m_vdCopy[m_readID]->data->positionBuffer[i]);
		velocity = (posCurrent - posLast) / fixedDelta;
		force = glm::vec3(0.0f);
		elasticity = m_simData.b_elMassCoeffs[i][0];
		mass = m_simData.b_elMassCoeffs[i][1];
		elDamp = m_simData.b_elMassCoeffs[i][2];
		airDamp = m_simData.b_elMassCoeffs[i][3];
		lockMultiplier = m_simData.b_multipliers[i][0];

		for (int j = 0; j < 4; ++j)
		{
			int nID = (int)(glm::roundEven(m_simData.b_neighbours[i][j]));
			if (nID < 0 || nID >= m_simData.m_vertexCount || nID == i)
			{
				//LOGI("%d", nID);
				continue;
			}
				
			nPos = glm::vec3(m_vd[m_readID]->data->positionBuffer[nID]);
			nPosLast = glm::vec3(m_vdCopy[m_readID]->data->positionBuffer[nID]);
			
			CalculateSpringForce(&posCurrent, &posLast, &nPos, &nPosLast, sls1[j],
				elasticity, elDamp, fixedDelta, &tempForce);

			force += tempForce * m_simData.b_neighbourMultipliers[i][j];
		}
		for (int j = 0; j < 4; ++j)
		{
			int nID = (int)(m_simData.b_neighboursDiag[i][j]);
			if (nID < 0 || nID >= m_simData.m_vertexCount || nID == i)
				continue;
			nPos = glm::vec3(m_vd[m_readID]->data->positionBuffer[nID]);
			nPosLast = glm::vec3(m_vdCopy[m_readID]->data->positionBuffer[nID]);

			CalculateSpringForce(&posCurrent, &posLast, &nPos, &nPosLast, sls2[j],
				elasticity, elDamp, fixedDelta, &tempForce);

			force += tempForce * m_simData.b_neighbourDiagMultipliers[i][j];
		}
		for (int j = 0; j < 4; ++j)
		{
			int nID = (int)(m_simData.b_neighbours2[i][j]);
			if (nID < 0 || nID >= m_simData.m_vertexCount || nID == i)
				continue;
			nPos = glm::vec3(m_vd[m_readID]->data->positionBuffer[nID]);
			nPosLast = glm::vec3(m_vdCopy[m_readID]->data->positionBuffer[nID]);

			CalculateSpringForce(&posCurrent, &posLast, &nPos, &nPosLast, sls3[j],
				elasticity, elDamp, fixedDelta, &tempForce);

			force += tempForce * m_simData.b_neighbour2Multipliers[i][j];
		}

		force += mass * glm::vec3(0.0f, -gravity * 0.1f, 0.0f);
		force += (-airDamp * velocity);
		force *= lockMultiplier;
		force = force / mass;

		posNew = 2.0f * posCurrent - posLast + force * fixedDelta * fixedDelta;

		m_vdCopy[m_writeID]->data->positionBuffer[i] = glm::vec4(posCurrent, 1.0f);
		m_vd[m_writeID]->data->positionBuffer[i] = glm::vec4(posNew, 1.0f);
	}

	return err;
}

inline unsigned int ClothSimulator::UpdateSimPBCPU(float gravity, float fixedDelta)
{
	unsigned int err = CS_ERR_NONE;
	glm::vec3 posCurrent, posLast, velocity, force, posPredicted, cPos, posNew;
	float elasticity, airDamp, mass, lockMultiplier;
	float elBias = 0.0005f;
	glm::vec4 sls1 =
	{
		m_simData.c_springLengths.y,
		m_simData.c_springLengths.x,
		m_simData.c_springLengths.y,
		m_simData.c_springLengths.x
	};
	glm::vec4 sls2 =
	{
		m_simData.c_springLengths.z,
		m_simData.c_springLengths.z,
		m_simData.c_springLengths.z,
		m_simData.c_springLengths.z
	};
	glm::vec4 sls3 =
	{
		m_simData.c_springLengths.y * m_simData.c_springLengths.w,
		m_simData.c_springLengths.x * m_simData.c_springLengths.w,
		m_simData.c_springLengths.y * m_simData.c_springLengths.w,
		m_simData.c_springLengths.x * m_simData.c_springLengths.w
	};

	for (int i = 0; i < m_simData.m_vertexCount; ++i)
	{
		posCurrent = glm::vec3(m_vd[m_readID]->data->positionBuffer[i]);
		posLast = glm::vec3(m_vdCopy[m_readID]->data->positionBuffer[i]);
		velocity = (posCurrent - posLast) / fixedDelta;
		force = glm::vec3(0.0f, 0.0f, 0.0f);
		elasticity = m_simData.b_elMassCoeffs[i].x * elBias;
		mass = m_simData.b_elMassCoeffs[i].y;
		airDamp = m_simData.b_elMassCoeffs[i].w;
		lockMultiplier = m_simData.b_multipliers[i].x;

		force += mass * glm::vec3(0.0f, -gravity * 0.1f, 0.0f);
		force += -airDamp * velocity;
		force *= lockMultiplier;
		force = force / mass;
		posPredicted = 2.0f * posCurrent - posLast + force * fixedDelta * fixedDelta;

		cPos = glm::vec3(0.0f, 0.0f, 0.0f);
		for (int j = 0; j < 4; ++j)
		{
			int nID = (int)(glm::roundEven(m_simData.b_neighbours[i][j]));
			if (nID < 0 || nID >= m_simData.m_vertexCount || nID == i)
			{
				continue;
			}

			glm::vec3 nPos = glm::vec3(m_vd[m_readID]->data->positionBuffer[nID]);
			glm::vec3 cstr = glm::vec3(0.0f, 0.0f, 0.0f);
			float w = 1.0f;
			CalcDistConstraint(&posCurrent, &nPos, mass, sls1[j], elasticity, &cstr, &w);
			cPos -= cstr * w * m_simData.b_neighbourMultipliers[i][j];
		}
		for (int j = 0; j < 4; ++j)
		{
			int nID = (int)(glm::roundEven(m_simData.b_neighboursDiag[i][j]));
			if (nID < 0 || nID >= m_simData.m_vertexCount || nID == i)
			{
				continue;
			}

			glm::vec3 nPos = glm::vec3(m_vd[m_readID]->data->positionBuffer[nID]);
			glm::vec3 cstr = glm::vec3(0.0f, 0.0f, 0.0f);
			float w = 1.0f;
			CalcDistConstraint(&posCurrent, &nPos, mass, sls2[j], elasticity, &cstr, &w);
			cPos -= cstr * w * m_simData.b_neighbourDiagMultipliers[i][j];
		}
		for (int j = 0; j < 4; ++j)
		{
			int nID = (int)(glm::roundEven(m_simData.b_neighbours2[i][j]));
			if (nID < 0 || nID >= m_simData.m_vertexCount || nID == i)
			{
				continue;
			}

			glm::vec3 nPos = glm::vec3(m_vd[m_readID]->data->positionBuffer[nID]);
			glm::vec3 cstr = glm::vec3(0.0f, 0.0f, 0.0f);
			float w = 1.0f;
			CalcDistConstraint(&posCurrent, &nPos, mass, sls3[j], elasticity, &cstr, &w);
			cPos -= cstr * w * m_simData.b_neighbour2Multipliers[i][j];
		}

		posNew = posPredicted + cPos * lockMultiplier;
		m_vd[m_writeID]->data->positionBuffer[i] = glm::vec4(posNew, 1.0f);
		m_vdCopy[m_writeID]->data->positionBuffer[i] = glm::vec4(posCurrent, 1.0f);
	}
	
	return err;
}

/////////////////////////////////

inline void ClothSimulator::CalculateSpringForce
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
)
{
	glm::vec3 mVelocity = (*mPos - *mPosLast) / fixedDelta;
	glm::vec3 nVelocity = (*nPos - *nPosLast) / fixedDelta;

	glm::vec3 f = *mPos - *nPos;
	glm::vec3 n = glm::normalize(f);
	float fLength = glm::length(f);
	float spring = fLength - sLength;
	glm::vec3 spr = -elCoeff * spring * n;

	glm::vec3 dV = mVelocity - nVelocity;
	float damp = dampCoeff * (glm::dot(dV, f) / fLength);

	float sL = length(spr);
	glm::vec3 dmp = n * glm::min(sL, damp);

	glm::vec3 fin = (spr + dmp);
	ret->x = fin.x;
	ret->y = fin.y;
	ret->z = fin.z;
}

inline void ClothSimulator::CalcDistConstraint(
	glm::vec3* mPos,
	glm::vec3* nPos,
	float mass,
	float sLength,
	float elCoeff,
	glm::vec3* outConstraint,
	float* outW
	) 
{
	glm::vec3 diff = *mPos - *nPos;
	float cLength = length(diff);
	glm::vec3 dP = (2.0f * mass) * (cLength - sLength) * (diff / cLength) * elCoeff;
	*outConstraint = dP;
	*outW = 1.0f / mass;
}

inline void ClothSimulator::CopyVertexData(VertexData * source, VertexData * dest)
{
	if (dest->data == nullptr)
		dest->data = new VertexDataRaw;
	if (dest->ids == nullptr)
		dest->ids = new VertexDataID;

	dest->data->vertexCount = source->data->vertexCount;
	dest->data->indexCount = source->data->indexCount;

	if (dest->data->barycentricBuffer == nullptr)
		dest->data->barycentricBuffer = new glm::vec4[dest->data->indexCount];
	if (dest->data->colorBuffer == nullptr)
		dest->data->colorBuffer = new glm::vec4[dest->data->vertexCount];
	if (dest->data->indexBuffer == nullptr)
		dest->data->indexBuffer = new unsigned int[dest->data->indexCount];
	if (dest->data->normalBuffer == nullptr)
		dest->data->normalBuffer = new glm::vec4[dest->data->vertexCount];
	if (dest->data->positionBuffer == nullptr)
		dest->data->positionBuffer = new glm::vec4[dest->data->vertexCount];
	if (dest->data->uvBuffer == nullptr)
		dest->data->uvBuffer = new glm::vec2[dest->data->vertexCount];

	dest->ids->barycentricBuffer = source->ids->barycentricBuffer;
	dest->ids->colorBuffer = source->ids->colorBuffer;
	dest->ids->indexBuffer = source->ids->indexBuffer;
	dest->ids->normalBuffer = source->ids->normalBuffer;
	dest->ids->uvBuffer = source->ids->uvBuffer;
	dest->ids->vertexArrayID = source->ids->vertexArrayID;
	dest->ids->vertexBuffer = source->ids->vertexBuffer;	

	for (unsigned int i = 0; i < dest->data->vertexCount; ++i)
	{
		dest->data->positionBuffer[i] = source->data->positionBuffer[i];
		dest->data->uvBuffer[i] = source->data->uvBuffer[i];
		dest->data->normalBuffer[i] = source->data->normalBuffer[i];
		dest->data->colorBuffer[i] = source->data->colorBuffer[i];
	}

	for (unsigned int i = 0; i < dest->data->indexCount; ++i)
	{
		dest->data->indexBuffer[i] = source->data->indexBuffer[i];
		dest->data->barycentricBuffer[i] = source->data->barycentricBuffer[i];
	}
}

inline void ClothSimulator::SwapRWIds()
{
	unsigned int tmp = m_readID;
	m_readID = m_writeID;
	m_writeID = tmp;
}

inline void ClothSimulator::AppendRestart()
{
	m_ifRestart = true;
}

inline void ClothSimulator::RestartSimulation()
{
	// update simulation parameters from UI

	// update data buffers on GPU with initial values

	if(m_readID == 1)
		m_meshPlane->SwapDataPtrs();

	m_readID = 0;
	m_writeID = 1;
	//for (uint i = 0; i < 2; ++i)
	//{
	//	glBindBuffer(GL_ARRAY_BUFFER, m_vboPosID[i]);
	//	glBufferSubData(GL_ARRAY_BUFFER, 0,
	//		m_vdCopy->data->vertexCount * sizeof(m_vdCopy->data->positionBuffer[0]),
	//		&m_vdCopy->data->positionBuffer[0].x);

	//	glBindBuffer(GL_ARRAY_BUFFER, m_vboPosLastID[i]);
	//	glBufferSubData(GL_ARRAY_BUFFER, 0,
	//		m_simData.m_vertexCount * sizeof(m_vdCopy->data->positionBuffer[0]),
	//		&m_vdCopy->data->positionBuffer[0].x);

	//	glBindBuffer(GL_ARRAY_BUFFER, m_vboNrmID[i]);
	//	glBufferSubData(GL_ARRAY_BUFFER, 0,
	//		m_vdCopy->data->vertexCount * sizeof(m_vdCopy->data->normalBuffer[0]),
	//		m_vdCopy->data->normalBuffer);
	//}
	//glBindBuffer(GL_ARRAY_BUFFER, 0);



	Shutdown();
	Initialize();
}
