#include "MeshGL.h"


MeshGL::MeshGL(SimObject* obj) : Mesh(obj)
{
	m_vertexArray = nullptr;
	m_vertexData = nullptr;
}

MeshGL::MeshGL(const MeshGL* m) : Mesh(m)
{
}


MeshGL::~MeshGL()
{
}

unsigned int MeshGL::Initialize()
{
	m_vertexData = new VertexData;
	m_vertexData->data = new VertexDataRaw;
	m_vertexData->ids = new VertexDataID;

	// generate vertex data and vertex array
	GenerateVertexData();

	// setting up buffers

	glGenVertexArrays(1, &m_vertexData->ids->vertexArrayID);
	glBindVertexArray(m_vertexData->ids->vertexArrayID);

	glGenBuffers(1, &m_vertexData->ids->vertexBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, m_vertexData->ids->vertexBuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(m_vertexData->data->positionBuffer[0]) * m_vertexData->data->vertexCount, 
		m_vertexData->data->positionBuffer, GL_STATIC_DRAW);

	glGenBuffers(1, &m_vertexData->ids->uvBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, m_vertexData->ids->uvBuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(m_vertexData->data->uvBuffer[0]) * m_vertexData->data->vertexCount,
		m_vertexData->data->uvBuffer, GL_STATIC_DRAW);

	glGenBuffers(1, &m_vertexData->ids->normalBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, m_vertexData->ids->normalBuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(m_vertexData->data->normalBuffer[0]) * m_vertexData->data->vertexCount,
		m_vertexData->data->normalBuffer, GL_STATIC_DRAW);

	glGenBuffers(1, &m_vertexData->ids->colorBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, m_vertexData->ids->colorBuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(m_vertexData->data->colorBuffer[0]) * m_vertexData->data->vertexCount,
		m_vertexData->data->colorBuffer, GL_STATIC_DRAW);

	glGenBuffers(1, &m_vertexData->ids->indexBuffer);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_vertexData->ids->indexBuffer);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(m_vertexData->data->indexBuffer[0]) * m_vertexData->data->indexCount,
		m_vertexData->data->indexBuffer, GL_STATIC_DRAW);

	// obtaining IDs of Renderer's current shader
	UpdateShaderIDs(Renderer::GetInstance()->GetCurrentShaderID());

	return CS_ERR_NONE;
}

unsigned int MeshGL::Shutdown()
{
	// cleaning up for openGL
	glDeleteVertexArrays(1, &m_vertexData->ids->vertexArrayID);
	glDeleteBuffers(1, &m_vertexData->ids->vertexBuffer);
	glDeleteBuffers(1, &m_vertexData->ids->uvBuffer);
	glDeleteBuffers(1, &m_vertexData->ids->normalBuffer);
	glDeleteBuffers(1, &m_vertexData->ids->colorBuffer);
	glDeleteBuffers(1, &m_vertexData->ids->indexBuffer);

	if (m_vertexData != nullptr)
		delete m_vertexData;

	if (m_vertexArray != nullptr)
		delete m_vertexArray;

	return CS_ERR_NONE;
}

unsigned int MeshGL::Draw()
{
	glm::mat4 wvp = (*System::GetInstance()->GetCurrentScene()->GetCamera()->GetViewProjMatrix()) * 
		(* m_obj->GetTransform()->GetWorldMatrix());

	glUniformMatrix4fv(id_worldViewProj, 1, GL_FALSE, &wvp[0][0]);
	glUniformMatrix4fv(id_world, 1, GL_FALSE, &(*m_obj->GetTransform()->GetWorldMatrix())[0][0]);
	glUniformMatrix4fv(id_worldInvTrans, 1, GL_FALSE, &(*m_obj->GetTransform()->GetWorldInverseTransposeMatrix())[0][0]);

	glm::vec3* tempEye = System::GetInstance()->GetCurrentScene()->GetCamera()->GetDirection();
	glUniform4f(id_eyeVector, -tempEye->x, -tempEye->y, -tempEye->z, 1.0f);

	// here we will set up light from global lighting in the scene
	if (System::GetInstance()->GetCurrentScene()->GetAmbientLight() != nullptr)
	{
		LightAmbient* amb = System::GetInstance()->GetCurrentScene()->GetAmbientLight();
		glm::vec3* diff = amb->GetDiffuseColor();
		glUniform4f(id_lightAmb, diff->x, diff->y, diff->z, 1.0f);
	}

	if (System::GetInstance()->GetCurrentScene()->GetLightDirectionalCount() > 0 )
	{
		LightDirectional* firstDiff = System::GetInstance()->GetCurrentScene()->GetLightDirectional(0);
		if (firstDiff != nullptr)
		{
			glm::vec3 *d, *s;
			d = firstDiff->GetDiffuseColor();
			s = firstDiff->GetSpecularColor();
			glm::vec3* dir = firstDiff->GetDirection();
			glUniform4f(id_lightDiff, d->x, d->y, d->z, 1.0f);
			glUniform4f(id_lightSpec, s->x, s->y, s->z, 1.0f);
			glUniform4f(id_lightDir, dir->x, dir->y, dir->z, 1.0f);
		}
		
	}
	// here we will set up highlight?
	// here we will set up glossiness
	glUniform1f(id_gloss, 100.0f);

	// here we will set up texture?

	//////////////////////////////////////////

	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);
	glEnableVertexAttribArray(2);
	glEnableVertexAttribArray(3);

	glBindBuffer(GL_ARRAY_BUFFER, m_vertexData->ids->vertexBuffer);
	glVertexAttribPointer(
		0,
		3,
		GL_FLOAT,
		GL_FALSE,
		0,
		(void*)0
		);

	glBindBuffer(GL_ARRAY_BUFFER, m_vertexData->ids->uvBuffer);
	glVertexAttribPointer(
		1,
		2,
		GL_FLOAT,
		GL_FALSE,
		0,
		(void*)0
		);

	glBindBuffer(GL_ARRAY_BUFFER, m_vertexData->ids->normalBuffer);
	glVertexAttribPointer(
		2,
		3,
		GL_FLOAT,
		GL_FALSE,
		0,
		(void*)0
		);

	glBindBuffer(GL_ARRAY_BUFFER, m_vertexData->ids->colorBuffer);
	glVertexAttribPointer(
		3,
		4,
		GL_FLOAT,
		GL_FALSE,
		0,
		(void*)0
		);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_vertexData->ids->indexBuffer);


	glDrawElements(
		GL_TRIANGLES,
		m_vertexData->data->indexCount,
		GL_UNSIGNED_INT,
		(void*)0
		);


	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);
	glDisableVertexAttribArray(2);
	glDisableVertexAttribArray(3);

	return CS_ERR_NONE;
}

void MeshGL::UpdateShaderIDs(unsigned int shaderID)
{
	id_worldViewProj = glGetUniformLocation(shaderID, "WorldViewProj");
	id_world = glGetUniformLocation(shaderID, "World");
	id_worldInvTrans = glGetUniformLocation(shaderID, "WorldInvTrans");
	id_eyeVector = glGetUniformLocation(shaderID, "EyeVector");
	id_lightDir = glGetUniformLocation(shaderID, "LightDir");
	id_lightDiff = glGetUniformLocation(shaderID, "LightDiff");
	id_lightSpec = glGetUniformLocation(shaderID, "LightSpec");
	id_lightAmb = glGetUniformLocation(shaderID, "LightAmb");
	id_gloss = glGetUniformLocation(shaderID, "Gloss");
}



void MeshGL::CreateVertexDataBuffers(unsigned int vCount, unsigned int iCount)
{
	m_vertexData->data->vertexCount = vCount;
	m_vertexData->data->indexCount = iCount;

	m_vertexData->data->positionBuffer = new glm::vec3[m_vertexData->data->vertexCount];
	m_vertexData->data->indexBuffer = new unsigned int[m_vertexData->data->indexCount];		// ?
	m_vertexData->data->uvBuffer = new glm::vec2[m_vertexData->data->vertexCount];
	m_vertexData->data->normalBuffer = new glm::vec3[m_vertexData->data->vertexCount];
	m_vertexData->data->colorBuffer = new glm::vec4[m_vertexData->data->vertexCount];
}