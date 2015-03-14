#include "Mesh.h"


Mesh::Mesh()
{
	m_vertexData = nullptr;
	m_vertexID = nullptr;
}


Mesh::~Mesh()
{
}

bool Mesh::Initialize(GLuint programID, Mesh* parent, string name, VertexData* data)
{
	this->parent = parent;
	m_name = name;

	m_vertexID = new VertexID;
	m_vertexData = data;

	glGenVertexArrays(1, &m_vertexID->vertexArrayID);
	glBindVertexArray(m_vertexID->vertexArrayID);

	modelMatrix = glm::mat4(1.0f);
	this->position = glm::vec3(0.0f, 0.0f, 0.0f);
	this->rotation = glm::vec3(0.0f, 0.0f, 0.0f);
	this->scale = glm::vec3(1.0f, 1.0f, 1.0f);

	glGenBuffers(1, &m_vertexID->vertexBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, m_vertexID->vertexBuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(m_vertexData->vertexPositionBuffer[0]) * 3 * m_vertexData->vertexCount, 
		m_vertexData->vertexPositionBuffer, GL_STATIC_DRAW);

	glGenBuffers(1, &m_vertexID->colorBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, m_vertexID->colorBuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(m_vertexData->vertexColorBuffer[0]) * 3 * m_vertexData->vertexCount,
		m_vertexData->vertexColorBuffer, GL_STATIC_DRAW);

	glGenBuffers(1, &m_vertexID->uvBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, m_vertexID->uvBuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(m_vertexData->vertexUVBuffer[0]) * 2 * m_vertexData->vertexCount,
		m_vertexData->vertexUVBuffer, GL_STATIC_DRAW);

	glGenBuffers(1, &m_vertexID->normalBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, m_vertexID->normalBuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(m_vertexData->vertexNormalBuffer[0]) * 3 * m_vertexData->vertexCount,
		m_vertexData->vertexNormalBuffer, GL_STATIC_DRAW);

	glGenBuffers(1, &m_vertexID->indexBuffer);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_vertexID->indexBuffer);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(m_vertexData->indexBuffer[0]) * m_vertexData->indexCount,
		m_vertexData->indexBuffer, GL_STATIC_DRAW);

	mvpMatrixID = glGetUniformLocation(programID, "mvpMatrix");
	modelID = glGetUniformLocation(programID, "model");

	return true;
}

void Mesh::Shutdown()
{
	if (children.size() > 0)
	{
		for (vector<Mesh*>::iterator it = children.begin(); it != children.end(); ++it)
		{
			(*it)->Shutdown();
			delete (*it);
		}
	}
	if (m_vertexData != nullptr)
	{
		if (m_vertexData->vertexPositionBuffer != nullptr)
			delete[] m_vertexData->vertexPositionBuffer;
		if (m_vertexData->vertexColorBuffer != nullptr)
			delete[] m_vertexData->vertexColorBuffer;
		if (m_vertexData->vertexNormalBuffer != nullptr)
			delete[] m_vertexData->vertexNormalBuffer;
		if (m_vertexData->vertexUVBuffer != nullptr)
			delete[] m_vertexData->vertexUVBuffer;

		delete m_vertexData;
	}
	if (m_texture != nullptr)
	{
		m_texture->Shutdown();
		delete m_texture;
	}

	if (m_vertexID != nullptr)
		delete m_vertexID;
	

	glDeleteBuffers(1, &m_vertexID->vertexBuffer);
	glDeleteVertexArrays(1, &m_vertexID->vertexArrayID);
	glDeleteBuffers(1, &m_vertexID->colorBuffer);
	glDeleteBuffers(1, &m_vertexID->uvBuffer);
	glDeleteBuffers(1, &m_vertexID->normalBuffer);
}

void Mesh::Draw(glm::mat4* projectionMatrix, glm::mat4* viewMatrix, glm::vec3* eyeVector, GLuint eyeVectorID, Light* light)
{
	mvpMatrix = (*projectionMatrix) * (*viewMatrix) * modelMatrix;
	glm::vec4 temp = glm::normalize(light->lightDirection * modelMatrix);
	glm::vec4 tempEye = glm::normalize(glm::vec4(*eyeVector, 1.0f) * (modelMatrix));
	glUniformMatrix4fv(mvpMatrixID, 1, GL_FALSE, &mvpMatrix[0][0]);
	glUniformMatrix4fv(modelID, 1, GL_FALSE, &modelMatrix[0][0]);
	glUniform4f(eyeVectorID, tempEye.x, tempEye.y, tempEye.z, tempEye.w);
	glUniform4f(light->lightDirID, temp.x, temp.y, temp.z, temp.w);
	glUniform4f(light->lightDifID, light->lightDiffuse.x, light->lightDiffuse.y, light->lightDiffuse.z, light->lightDiffuse.w);
	glUniform4f(light->lightSpecID, light->lightSpecular.x, light->lightSpecular.y, light->lightSpecular.z, light->lightSpecular.w);
	glUniform4f(light->lightAmbID, light->lightAmbient.x, light->lightAmbient.y, light->lightAmbient.z, light->lightAmbient.w);
	glUniform1f(light->glossID, light->glossiness);

	glBindTexture(GL_TEXTURE_2D, m_texture->GetID());

	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);
	glEnableVertexAttribArray(2);
	glEnableVertexAttribArray(3);

	glBindBuffer(GL_ARRAY_BUFFER, m_vertexID->vertexBuffer);
	glVertexAttribPointer(
		0,
		3,
		GL_FLOAT,
		GL_FALSE,
		0,
		(void*)0
		);

	glBindBuffer(GL_ARRAY_BUFFER, m_vertexID->colorBuffer);
	glVertexAttribPointer(
		1,
		3,
		GL_FLOAT,
		GL_FALSE,
		0,
		(void*)0
		);

	glBindBuffer(GL_ARRAY_BUFFER, m_vertexID->uvBuffer);
	glVertexAttribPointer(
		2,
		2,
		GL_FLOAT,
		GL_FALSE,
		0,
		(void*)0
		);

	glBindBuffer(GL_ARRAY_BUFFER, m_vertexID->normalBuffer);
	glVertexAttribPointer(
		3,
		3,
		GL_FLOAT,
		GL_FALSE,
		0,
		(void*)0
		);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_vertexID->indexBuffer);
	
	glDrawElements
		(
		GL_TRIANGLES,
		m_vertexData->indexCount,
		GL_UNSIGNED_INT, 
		(void*)0
	);

	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);
	glDisableVertexAttribArray(2);
	glDisableVertexAttribArray(3);
	
	for (vector<Mesh*>::iterator it = children.begin(); it != children.end(); ++it)
	{
		(*it)->Draw(projectionMatrix, viewMatrix, eyeVector, eyeVectorID, light);
	}
}

void Mesh::Transform(const glm::vec3* position, const glm::vec3* rotation, const glm::vec3* scale_v)
{
	this->position = *position;
	this->rotation = *rotation;
	this->scale = *scale_v;


	glm::mat4 translation, rotation_x, rotation_y, rotation_z, scale_m, parentMatrix;
	translation = glm::translate(this->position);
	rotation_x = glm::rotate(this->rotation.x, glm::vec3(1.0f, 0.0f, 0.0f));
	rotation_y = glm::rotate(this->rotation.y, glm::vec3(0.0f, 1.0f, 0.0f));
	rotation_z = glm::rotate(this->rotation.z, glm::vec3(0.0f, 0.0f, 1.0f));
	scale_m = glm::scale(this->scale);

	(parent != NULL) ? (parentMatrix = *parent->GetModelMatrix()) : (parentMatrix = glm::mat4(1.0f));

	modelMatrix = parentMatrix*translation*(rotation_x*rotation_y*rotation_z)*scale_m;

	for (vector<Mesh*>::iterator it = children.begin(); it != children.end(); ++it)
	{
		(*it)->Transform((*it)->GetPosition(), (*it)->GetRotation(), (*it)->GetScale());
	}
}

void Mesh::AddChild(Mesh* child)
{
	children.push_back(child);
	child->SetParent(this);
}

vector<Mesh*>* Mesh::GetChildren()
{
	return &children;
}

Mesh* Mesh::GetParent()
{
	return parent;
}

void Mesh::SetParent(Mesh* parent)
{
	this->parent = parent;
}

glm::mat4* Mesh::GetModelMatrix()
{
	return &modelMatrix;
}

glm::vec3* Mesh::GetPosition()
{
	return &position;
}

glm::vec3* Mesh::GetRotation()
{
	return &rotation;
}

glm::vec3* Mesh::GetScale()
{
	return &scale;
}

void Mesh::SetTexture(Texture* texture)
{
	m_texture = texture;
}
