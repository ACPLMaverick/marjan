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

	m_readID = 0;
	m_writeID = 1;

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
		glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, 0);

		glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, m_vboPosLastID[i]);
		glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER,
			m_simData->m_vertexCount * sizeof(m_simData->b_positionLast[0]),
			&m_simData->b_positionLast[0].x,
			GL_DYNAMIC_COPY);
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, 0);



		glBindBuffer(GL_ARRAY_BUFFER, m_vd[0]->ids->normalBuffer);
		glEnableVertexAttribArray(2);
		glVertexAttribPointer(2, 4, GL_UNSIGNED_INT, GL_FALSE, 0, 0);

		if (i == 0)
			glGenBuffers(1, &m_simData->i_neighbours);

		glBindBuffer(GL_ARRAY_BUFFER, m_simData->i_neighbours);
		glBufferData(GL_ARRAY_BUFFER,
			m_simData->m_vertexCount * 4 * sizeof(m_simData->b_neighbours[0]),
			m_simData->b_neighbours,
			GL_STATIC_DRAW);
		glEnableVertexAttribArray(3);
		glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, 0, 0);

		if (i == 0)
			glGenBuffers(1, &m_simData->i_neighbourMultipliers);
		glBindBuffer(GL_ARRAY_BUFFER, m_simData->i_neighbourMultipliers);
		glBufferData(GL_ARRAY_BUFFER,
			m_simData->m_vertexCount * 4 * sizeof(m_simData->b_neighbourMultipliers[0]),
			m_simData->b_neighbourMultipliers,
			GL_STATIC_DRAW);
		glEnableVertexAttribArray(4);
		glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, 0, 0);

		if (i == 0)
			glGenBuffers(1, &m_simData->i_springLengths);
		glBindBuffer(GL_ARRAY_BUFFER, m_simData->i_springLengths);
		glBufferData(GL_ARRAY_BUFFER,
			m_simData->m_vertexCount * 4 * sizeof(m_simData->b_springLengths[0]),
			m_simData->b_springLengths,
			GL_STATIC_DRAW);
		glEnableVertexAttribArray(5);
		glVertexAttribPointer(5, 4, GL_FLOAT, GL_FALSE, 0, 0);

		if (i == 0)
			glGenBuffers(1, &m_simData->i_elasticity);
		glBindBuffer(GL_ARRAY_BUFFER, m_simData->i_elasticity);
		glBufferData(GL_ARRAY_BUFFER,
			m_simData->m_vertexCount * sizeof(m_simData->b_elasticity[0]),
			m_simData->b_elasticity,
			GL_STATIC_DRAW);
		glEnableVertexAttribArray(6);
		glVertexAttribPointer(6, 1, GL_FLOAT, GL_FALSE, 0, 0);

		if (i == 0)
			glGenBuffers(1, &m_simData->i_mass);
		glBindBuffer(GL_ARRAY_BUFFER, m_simData->i_mass);
		glBufferData(GL_ARRAY_BUFFER,
			m_simData->m_vertexCount * sizeof(m_simData->b_mass[0]),
			m_simData->b_mass,
			GL_STATIC_DRAW);
		glEnableVertexAttribArray(7);
		glVertexAttribPointer(7, 1, GL_FLOAT, GL_FALSE, 0, 0);

		if (i == 0)
			glGenBuffers(1, &m_simData->i_dampCoeff);
		glBindBuffer(GL_ARRAY_BUFFER, m_simData->i_dampCoeff);
		glBufferData(GL_ARRAY_BUFFER,
			m_simData->m_vertexCount * sizeof(m_simData->b_dampCoeff[0]),
			m_simData->b_dampCoeff,
			GL_STATIC_DRAW);
		glEnableVertexAttribArray(8);
		glVertexAttribPointer(8, 1, GL_FLOAT, GL_FALSE, 0, 0);

		if (i == 0)
			glGenBuffers(1, &m_simData->i_airDampCoeff);
		glBindBuffer(GL_ARRAY_BUFFER, m_simData->i_airDampCoeff);
		glBufferData(GL_ARRAY_BUFFER,
			m_simData->m_vertexCount * sizeof(m_simData->b_airDampCoeff[0]),
			m_simData->b_airDampCoeff,
			GL_STATIC_DRAW);
		glEnableVertexAttribArray(9);
		glVertexAttribPointer(9, 1, GL_FLOAT, GL_FALSE, 0, 0);

		if (i == 0)
			glGenBuffers(1, &m_simData->i_lockMultiplier);
		glBindBuffer(GL_ARRAY_BUFFER, m_simData->i_lockMultiplier);
		glBufferData(GL_ARRAY_BUFFER,
			m_simData->m_vertexCount * sizeof(m_simData->b_lockMultiplier[0]),
			m_simData->b_lockMultiplier,
			GL_STATIC_DRAW);
		glEnableVertexAttribArray(10);
		glVertexAttribPointer(10, 1, GL_FLOAT, GL_FALSE, 0, 0);

		if (i == 0)
			glGenBuffers(1, &m_simData->i_colliderMultiplier);
		glBindBuffer(GL_ARRAY_BUFFER, m_simData->i_colliderMultiplier);
		glBufferData(GL_ARRAY_BUFFER,
			m_simData->m_vertexCount * sizeof(m_simData->b_colliderMultiplier[0]),
			m_simData->b_colliderMultiplier,
			GL_STATIC_DRAW);
		glEnableVertexAttribArray(11);
		glVertexAttribPointer(11, 1, GL_FLOAT, GL_FALSE, 0, 0);
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

	m_kernelID->uniformIDs->push_back(glGetUniformLocation(m_kernelID->id, "VertexCount"));
	m_kernelID->uniformIDs->push_back(glGetUniformLocation(m_kernelID->id, "EdgesWidthAll"));
	m_kernelID->uniformIDs->push_back(glGetUniformLocation(m_kernelID->id, "EdgesLengthAll"));
	m_kernelID->uniformIDs->push_back(glGetUniformLocation(m_kernelID->id, "DeltaTime"));
	m_kernelID->uniformIDs->push_back(glGetUniformLocation(m_kernelID->id, "Gravity"));

	return err;
}

inline unsigned int ClothSimulatorMSGPU::ShutdownSim()
{
	unsigned int err = CS_ERR_NONE;

	// shutdown kernels 
	glDeleteVertexArrays(2, m_vaoUpdateID);
	glDeleteBuffers(2, m_vboPosLastID);

	glDeleteBuffers(1, &m_simData->i_neighbours);
	glDeleteBuffers(1, &m_simData->i_neighbourMultipliers);
	glDeleteBuffers(1, &m_simData->i_springLengths);
	glDeleteBuffers(1, &m_simData->i_elasticity);
	glDeleteBuffers(1, &m_simData->i_mass);
	glDeleteBuffers(1, &m_simData->i_dampCoeff);
	glDeleteBuffers(1, &m_simData->i_airDampCoeff);
	glDeleteBuffers(1, &m_simData->i_lockMultiplier);
	glDeleteBuffers(1, &m_simData->i_colliderMultiplier);

	// shutdown method-related data

	return err;
}

inline unsigned int ClothSimulatorMSGPU::UpdateSim(float gravity, float fixedDelta, BoxAAData * bColliders, SphereData * sColliders, glm::mat4 * wm)
{
	unsigned int err = CS_ERR_NONE;

	ShaderID* cShaderID = Renderer::GetInstance()->GetCurrentShaderID();
	glUseProgram(m_kernelID->id);

	glUniform1i(m_kernelID->uniformIDs->at(0), m_simData->m_vertexCount);
	glUniform1i(m_kernelID->uniformIDs->at(1), m_simData->m_edgesWidthAll);
	glUniform1i(m_kernelID->uniformIDs->at(2), m_simData->m_edgesLengthAll);
	glUniform1f(m_kernelID->uniformIDs->at(3), fixedDelta);
	glUniform1f(m_kernelID->uniformIDs->at(4), gravity);

	glBindBufferBase(GL_UNIFORM_BUFFER, m_texPosID[m_readID], m_vboPosID[m_readID]);
	glBindBufferBase(GL_UNIFORM_BUFFER, m_texPosLastID[m_readID], m_vboPosLastID[m_readID]);


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
	glBindBuffer(GL_ARRAY_BUFFER, m_simData->i_neighbours);
	glEnableVertexAttribArray(3);
	glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, 0, 0);
	glBindBuffer(GL_ARRAY_BUFFER, m_simData->i_neighbourMultipliers);
	glEnableVertexAttribArray(4);
	glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, 0, 0);
	glBindBuffer(GL_ARRAY_BUFFER, m_simData->i_springLengths);
	glEnableVertexAttribArray(5);
	glVertexAttribPointer(5, 4, GL_FLOAT, GL_FALSE, 0, 0);
	glBindBuffer(GL_ARRAY_BUFFER, m_simData->i_elasticity);
	glEnableVertexAttribArray(6);
	glVertexAttribPointer(6, 1, GL_FLOAT, GL_FALSE, 0, 0);
	glBindBuffer(GL_ARRAY_BUFFER, m_simData->i_mass);
	glEnableVertexAttribArray(7);
	glVertexAttribPointer(7, 1, GL_FLOAT, GL_FALSE, 0, 0);
	glBindBuffer(GL_ARRAY_BUFFER, m_simData->i_dampCoeff);
	glEnableVertexAttribArray(8);
	glVertexAttribPointer(8, 1, GL_FLOAT, GL_FALSE, 0, 0);
	glBindBuffer(GL_ARRAY_BUFFER, m_simData->i_airDampCoeff);
	glEnableVertexAttribArray(9);
	glVertexAttribPointer(9, 1, GL_FLOAT, GL_FALSE, 0, 0);
	glBindBuffer(GL_ARRAY_BUFFER, m_simData->i_lockMultiplier);
	glEnableVertexAttribArray(10);
	glVertexAttribPointer(10, 1, GL_FLOAT, GL_FALSE, 0, 0);
	glBindBuffer(GL_ARRAY_BUFFER, m_simData->i_colliderMultiplier);
	glEnableVertexAttribArray(11);
	glVertexAttribPointer(11, 1, GL_FLOAT, GL_FALSE, 0, 0);

	glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, m_vboPosID[m_writeID]);
	glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 1, m_vboPosLastID[m_writeID]);


	glEnable(GL_RASTERIZER_DISCARD);

	glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, m_tfID);
	glBeginTransformFeedback(GL_POINTS);
	glDrawArrays(GL_POINTS, 0, m_simData->m_vertexCount);
	glEndTransformFeedback();

	glFlush();
	glDisable(GL_RASTERIZER_DISCARD);

	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);
	glDisableVertexAttribArray(2);
	glDisableVertexAttribArray(3);
	glDisableVertexAttribArray(4);
	glDisableVertexAttribArray(5);
	glDisableVertexAttribArray(6);
	glDisableVertexAttribArray(7);
	glDisableVertexAttribArray(8);
	glDisableVertexAttribArray(9);
	glDisableVertexAttribArray(10);
	glDisableVertexAttribArray(11);
	/*
	glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, m_vboPosLastID[m_writeID]);
	glm::vec4* ptr = (glm::vec4*)glMapBufferRange(GL_TRANSFORM_FEEDBACK_BUFFER, 0, m_simData->m_vertexCount, GL_MAP_READ_BIT);
	for (int i = 0; i < m_simData->m_vertexCount; ++i, ++ptr)
	LOGI("%f %f %f %f", ptr->x, ptr->y, ptr->z, ptr->w);
	glUnmapBuffer(GL_TRANSFORM_FEEDBACK_BUFFER);
	*/
	SwapRWIds();
	m_meshPlane->SwapDataPtrs();

	glUseProgram(cShaderID->id);

	return err;
}

inline void ClothSimulatorMSGPU::SwapRWIds()
{
	unsigned int tmp = m_readID;
	m_readID = m_writeID;
	m_writeID = tmp;
}