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

	if (
		m_obj->GetMesh(0) != nullptr
		)
	{
		m_meshPlane = (MeshGLPlane*)m_obj->GetMesh(0);
	}
	else return CS_ERR_CLOTHSIMULATOR_MESH_OBTAINING_ERROR;

	m_readID = 0;
	m_writeID = 1;

	// get a copy of start data
	m_vd = m_meshPlane->GetVertexDataDualPtr();
	m_vdCopy = new VertexData;
	CopyVertexData(m_vd[0], m_vdCopy);

	// initialize simulation data
	m_simData = new SimData;
	m_simData->m_vertexCount = m_vd[0]->data->vertexCount;
	m_simData->m_edgesWidthAll = m_meshPlane->GetEdgesWidth();
	m_simData->m_edgesLengthAll = m_meshPlane->GetEdgesLength();

	m_simData->b_positionLast = new glm::vec3[m_simData->m_vertexCount];
	m_simData->b_springLengths = new float[m_simData->m_vertexCount * VERTEX_NEIGHBOURING_VERTICES];
	m_simData->b_neighbours = new unsigned int[m_simData->m_vertexCount * VERTEX_NEIGHBOURING_VERTICES];
	m_simData->b_neighbourMultipliers = new float[m_simData->m_vertexCount * VERTEX_NEIGHBOURING_VERTICES];
	m_simData->b_elasticity = new float[m_simData->m_vertexCount];
	m_simData->b_mass = new float[m_simData->m_vertexCount];
	m_simData->b_dampCoeff = new float[m_simData->m_vertexCount];
	m_simData->b_airDampCoeff = new float[m_simData->m_vertexCount];
	m_simData->b_lockMultiplier = new float[m_simData->m_vertexCount];
	m_simData->b_colliderMultiplier = new float[m_simData->m_vertexCount];

	glm::vec3 baseLength = glm::vec3(
		abs(m_vd[0]->data->positionBuffer[0].x - m_vd[0]->data->positionBuffer[m_simData->m_vertexCount - 1].x) / (float)(m_simData->m_edgesWidthAll - 1),
		0.0f,
		abs(m_vd[0]->data->positionBuffer[0].z - m_vd[0]->data->positionBuffer[m_simData->m_vertexCount - 1].z) / (float)(m_simData->m_edgesLengthAll - 1)
		);

	m_cellSize = glm::max(baseLength.x, baseLength.z) * 1.5f;

	for (unsigned int i = 0; i < m_simData->m_vertexCount; ++i)
	{
		m_simData->b_positionLast[i] = m_vd[0]->data->positionBuffer[i];
		m_simData->b_mass[i] = VERTEX_MASS;
		m_simData->b_airDampCoeff[i] = VERTEX_AIR_DAMP;
		m_simData->b_lockMultiplier[i] = 1.0f;
		m_simData->b_colliderMultiplier[i] = VERTEX_COLLIDER_MULTIPLIER;
		m_simData->b_elasticity[i] = SPRING_ELASTICITY;
		m_simData->b_dampCoeff[i] = SPRING_ELASTICITY_DAMP;

		if (i < m_simData->m_edgesLengthAll ||
			i >= (m_simData->m_vertexCount - m_simData->m_edgesLengthAll) ||
			i % m_simData->m_edgesLengthAll == 0 ||
			i % m_simData->m_edgesLengthAll == (m_simData->m_edgesLengthAll - 1)
			)
		{
			m_simData->b_elasticity[i] *= SPRING_BORDER_MULTIPLIER;
			//m_simData->b_elasticity[i]Damp *= SPRING_BORDER_MULTIPLIER;
		}

		// calculating neighbouring vertices ids and spring lengths

		// upper
		m_simData->b_neighbours[i * VERTEX_NEIGHBOURING_VERTICES + 0] = (i - 1) % m_simData->m_vertexCount;
		if (i % m_simData->m_edgesLengthAll)
		{
			m_simData->b_neighbourMultipliers[i * VERTEX_NEIGHBOURING_VERTICES + 0] = 1.0f;
			m_simData->b_springLengths[i * VERTEX_NEIGHBOURING_VERTICES + 0] = baseLength.z;
		}
		else
		{
			m_simData->b_neighbourMultipliers[i * VERTEX_NEIGHBOURING_VERTICES + 0] = 0.0f;
			m_simData->b_springLengths[i * VERTEX_NEIGHBOURING_VERTICES + 0] = 0.0f;
		}

		// left
		m_simData->b_neighbours[i * VERTEX_NEIGHBOURING_VERTICES + 1] = (i - m_simData->m_edgesLengthAll) % m_simData->m_vertexCount;
		if (i >= m_simData->m_edgesLengthAll)
		{
			m_simData->b_neighbourMultipliers[i * VERTEX_NEIGHBOURING_VERTICES + 1] = 1.0f;
			m_simData->b_springLengths[i * VERTEX_NEIGHBOURING_VERTICES + 1] = baseLength.x;
		}
		else
		{
			m_simData->b_neighbourMultipliers[i * VERTEX_NEIGHBOURING_VERTICES + 1] = 0.0f;
			m_simData->b_springLengths[i * VERTEX_NEIGHBOURING_VERTICES + 1] = 0.0f;
		}

		// lower
		m_simData->b_neighbours[i * VERTEX_NEIGHBOURING_VERTICES + 2] = (i + 1) % m_simData->m_vertexCount;
		if (i % m_simData->m_edgesLengthAll != (m_simData->m_edgesLengthAll - 1))
		{
			m_simData->b_neighbourMultipliers[i * VERTEX_NEIGHBOURING_VERTICES + 2] = 1.0f;
			m_simData->b_springLengths[i * VERTEX_NEIGHBOURING_VERTICES + 2] = baseLength.z;
		}
		else
		{
			m_simData->b_neighbourMultipliers[i * VERTEX_NEIGHBOURING_VERTICES + 2] = 0.0f;
			m_simData->b_springLengths[i * VERTEX_NEIGHBOURING_VERTICES + 2] = 0.0f;
		}

		// right
		m_simData->b_neighbours[i * VERTEX_NEIGHBOURING_VERTICES + 3] = (i + m_simData->m_edgesLengthAll) % m_simData->m_vertexCount;
		if (i < (m_simData->m_vertexCount - m_simData->m_edgesLengthAll))
		{
			m_simData->b_neighbourMultipliers[i * VERTEX_NEIGHBOURING_VERTICES + 3] = 1.0f;
			m_simData->b_springLengths[i * VERTEX_NEIGHBOURING_VERTICES + 3] = baseLength.x;
		}
		else
		{
			m_simData->b_neighbourMultipliers[i * VERTEX_NEIGHBOURING_VERTICES + 3] = 0.0f;
			m_simData->b_springLengths[i * VERTEX_NEIGHBOURING_VERTICES + 3] = 0.0f;
		}
	}

	// hard-coded locks
	m_simData->b_lockMultiplier[0] = 0.0f;
	m_simData->b_lockMultiplier[(m_simData->m_vertexCount - m_simData->m_edgesLengthAll)] = 0.0f;
	
	/*
	int n;
	cl_platform_id* id;
	clGetPlatformIDs(2, id, 0);
	*/


	// OpenGL Data initialization and binding
	m_vaoRenderID[0] = m_vd[0]->ids->vertexArrayID;
	m_vaoRenderID[1] = m_vd[1]->ids->vertexArrayID;
	glGenVertexArrays(2, m_vaoUpdateID);
	m_vboPosID[0] = m_vd[0]->ids->vertexBuffer;
	m_vboPosID[1] = m_vd[1]->ids->vertexBuffer;
	glGenBuffers(2, m_vboPosLastID);

	
	for (uint i = 0; i < 2; ++i)
	{
		glBindVertexArray(m_vaoUpdateID[i]);
		glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, m_vboPosID[i]);
		glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER,
			m_vd[0]->data->vertexCount * sizeof(m_vd[0]->data->positionBuffer[0]),
			&m_vd[0]->data->positionBuffer[0].x,
			GL_DYNAMIC_COPY);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);

		glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, m_vboPosLastID[i]);
		glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER,
			m_simData->m_vertexCount * sizeof(m_simData->b_positionLast[0]),
			&m_simData->b_positionLast[0].x,
			GL_DYNAMIC_COPY);
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);
	}
	
	m_kernelID = ResourceManager::GetInstance()->LoadKernel(&KERNEL_SIM_NAME);
	glGenTransformFeedbacks(1, &m_tfID);
	glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, m_tfID);
	glTransformFeedbackVaryings(m_kernelID->id, KERNEL_SIM_OUTPUT_NAME_COUNT, KERNEL_SIM_OUTPUT_NAMES, GL_SEPARATE_ATTRIBS);
	glLinkProgram(m_kernelID->id);

	for (uint i = 0; i < 2; ++i)
	{
		m_texPosID[i] = glGetUniformBlockIndex(m_kernelID->id, KERNEL_SIM_INPUT_NAMES[0]);
		m_texPosLastID[i] = glGetUniformBlockIndex(m_kernelID->id, KERNEL_SIM_INPUT_NAMES[1]);


	}
	// Sim-specific initialization. GenTransformFeedbacks and shader loading goes here.
	err = InitializeSim();

	return err;
}

unsigned int ClothSimulator::Shutdown()
{
	unsigned int err = CS_ERR_NONE;

	// Sim-specific shutdown
	err = ShutdownSim();

	delete m_vdCopy;
	delete m_simData;

	return err;
}



unsigned int ClothSimulator::Update()
{
	unsigned int err = CS_ERR_NONE;

	VertexData* clothData = m_meshPlane->GetVertexDataPtr();
	BoxAAData* boxData = nullptr;
	SphereData* sphereData = nullptr;
	if (PhysicsManager::GetInstance()->GetBoxCollidersData()->size() != 0)
		boxData = &(PhysicsManager::GetInstance()->GetBoxCollidersData()->at(0));
	if (PhysicsManager::GetInstance()->GetSphereCollidersData()->size() != 0)
		sphereData = &(PhysicsManager::GetInstance()->GetSphereCollidersData()->at(0));

	glm::mat4 zero = glm::mat4();
	glm::mat4* wm = &zero;

	if (m_obj->GetTransform() != nullptr)
		wm = m_obj->GetTransform()->GetWorldMatrix();


	ShaderID* cShaderID = Renderer::GetInstance()->GetCurrentShaderID();
	glUseProgram(m_kernelID->id);

	glBindBufferBase(GL_UNIFORM_BUFFER, m_texPosID[m_readID], m_vboPosID[m_readID]);
	glBindBufferBase(GL_UNIFORM_BUFFER, m_texPosLastID[m_readID], m_vboPosLastID[m_readID]);


	glBindVertexArray(m_vaoUpdateID[m_readID]);
	glBindBuffer(GL_ARRAY_BUFFER, m_vboPosID[m_readID]);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
	glBindBuffer(GL_ARRAY_BUFFER, m_vboPosLastID[m_readID]);
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);

	glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, m_vboPosID[m_writeID]);
	glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 1, m_vboPosLastID[m_writeID]);
	

	glEnable(GL_RASTERIZER_DISCARD);

	glm::vec3* ptr = (glm::vec3*)glMapBufferRange(GL_TRANSFORM_FEEDBACK_BUFFER, 0, m_simData->m_vertexCount, GL_MAP_READ_BIT);
	for (int i = 0; i < m_simData->m_vertexCount; ++i, ++ptr)
		LOGI("%f %f %f", ptr->x, ptr->y, ptr->z);
	glUnmapBuffer(GL_TRANSFORM_FEEDBACK_BUFFER);

	glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, m_tfID);
	glBeginTransformFeedback(GL_POINTS);
	glDrawArrays(GL_POINTS, 0, m_simData->m_vertexCount);
	glEndTransformFeedback();

	glFlush();
	glDisable(GL_RASTERIZER_DISCARD);

	//ptr = (glm::vec3*)glMapBufferRange(GL_TRANSFORM_FEEDBACK_BUFFER, 0, m_simData->m_vertexCount, GL_MAP_READ_BIT);
	//glUnmapBuffer(GL_TRANSFORM_FEEDBACK_BUFFER);

	SwapRWIds();
	m_meshPlane->SwapDataPtrs();
	
	glUseProgram(cShaderID->id);

	err = UpdateSim(
		PhysicsManager::GetInstance()->GetGravity(),
		Timer::GetInstance()->GetFixedDeltaTime(),
		boxData,
		sphereData,
		wm
		);

	return err;
}
unsigned int ClothSimulator::Draw()
{
	unsigned int err = CS_ERR_NONE;

	return err;
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
		dest->data->barycentricBuffer = new glm::vec3[dest->data->indexCount];
	if (dest->data->colorBuffer == nullptr)
		dest->data->colorBuffer = new glm::vec4[dest->data->vertexCount];
	if (dest->data->indexBuffer == nullptr)
		dest->data->indexBuffer = new unsigned int[dest->data->indexCount];
	if (dest->data->normalBuffer == nullptr)
		dest->data->normalBuffer = new glm::vec3[dest->data->vertexCount];
	if (dest->data->positionBuffer == nullptr)
		dest->data->positionBuffer = new glm::vec3[dest->data->vertexCount];
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
