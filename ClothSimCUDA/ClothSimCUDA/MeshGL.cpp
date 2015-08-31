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