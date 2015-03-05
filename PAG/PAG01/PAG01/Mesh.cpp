#include "Mesh.h"


Mesh::Mesh()
{
	m_vertexData = nullptr;
}


Mesh::~Mesh()
{
}

bool Mesh::Initialize()
{
	glGenVertexArrays(1, &m_vertexArrayID);
	glBindVertexArray(m_vertexArrayID);

	modelMatrix = glm::mat4();
	this->position = glm::vec3(0.0f, 0.0f, 0.0f);
	this->rotation = glm::vec3(0.0f, 0.0f, 0.0f);
	this->scale = glm::vec3(1.0f, 1.0f, 1.0f);
 
	m_vertexData = new VertexData;
	m_vertexData->vertexColorBuffer = nullptr;
	m_vertexData->vertexNormalBuffer = nullptr;
	m_vertexData->vertexPositionBuffer = nullptr;
	m_vertexData->vertexUVBuffer = nullptr;
	m_vertexData->vertexCount = 12*3;
	m_vertexData->vertexPositionBuffer = new GLfloat[3*m_vertexData->vertexCount] 
	{
			-1.0f, -1.0f, -1.0f, // triangle 1 : begin
			-1.0f, -1.0f, 1.0f,
			-1.0f, 1.0f, 1.0f, // triangle 1 : end
			1.0f, 1.0f, -1.0f, // triangle 2 : begin
			-1.0f, -1.0f, -1.0f,
			-1.0f, 1.0f, -1.0f, // triangle 2 : end
			1.0f, -1.0f, 1.0f,
			-1.0f, -1.0f, -1.0f,
			1.0f, -1.0f, -1.0f,
			1.0f, 1.0f, -1.0f,
			1.0f, -1.0f, -1.0f,
			-1.0f, -1.0f, -1.0f,
			-1.0f, -1.0f, -1.0f,
			-1.0f, 1.0f, 1.0f,
			-1.0f, 1.0f, -1.0f,
			1.0f, -1.0f, 1.0f,
			-1.0f, -1.0f, 1.0f,
			-1.0f, -1.0f, -1.0f,
			-1.0f, 1.0f, 1.0f,
			-1.0f, -1.0f, 1.0f,
			1.0f, -1.0f, 1.0f,
			1.0f, 1.0f, 1.0f,
			1.0f, -1.0f, -1.0f,
			1.0f, 1.0f, -1.0f,
			1.0f, -1.0f, -1.0f,
			1.0f, 1.0f, 1.0f,
			1.0f, -1.0f, 1.0f,
			1.0f, 1.0f, 1.0f,
			1.0f, 1.0f, -1.0f,
			-1.0f, 1.0f, -1.0f,
			1.0f, 1.0f, 1.0f,
			-1.0f, 1.0f, -1.0f,
			-1.0f, 1.0f, 1.0f,
			1.0f, 1.0f, 1.0f,
			-1.0f, 1.0f, 1.0f,
			1.0f, -1.0f, 1.0f
	};

	m_vertexData->vertexColorBuffer = new GLfloat[3*m_vertexData->vertexCount]
	{
			0.583f, 0.771f, 0.014f,
			0.609f, 0.115f, 0.436f,
			0.327f, 0.483f, 0.844f,
			0.822f, 0.569f, 0.201f,
			0.435f, 0.602f, 0.223f,
			0.310f, 0.747f, 0.185f,
			0.597f, 0.770f, 0.761f,
			0.559f, 0.436f, 0.730f,
			0.359f, 0.583f, 0.152f,
			0.483f, 0.596f, 0.789f,
			0.559f, 0.861f, 0.639f,
			0.195f, 0.548f, 0.859f,
			0.014f, 0.184f, 0.576f,
			0.771f, 0.328f, 0.970f,
			0.406f, 0.615f, 0.116f,
			0.676f, 0.977f, 0.133f,
			0.971f, 0.572f, 0.833f,
			0.140f, 0.616f, 0.489f,
			0.997f, 0.513f, 0.064f,
			0.945f, 0.719f, 0.592f,
			0.543f, 0.021f, 0.978f,
			0.279f, 0.317f, 0.505f,
			0.167f, 0.620f, 0.077f,
			0.347f, 0.857f, 0.137f,
			0.055f, 0.953f, 0.042f,
			0.714f, 0.505f, 0.345f,
			0.783f, 0.290f, 0.734f,
			0.722f, 0.645f, 0.174f,
			0.302f, 0.455f, 0.848f,
			0.225f, 0.587f, 0.040f,
			0.517f, 0.713f, 0.338f,
			0.053f, 0.959f, 0.120f,
			0.393f, 0.621f, 0.362f,
			0.673f, 0.211f, 0.457f,
			0.820f, 0.883f, 0.371f,
			0.982f, 0.099f, 0.879f
	};

	glGenBuffers(1, &m_vertexBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, m_vertexBuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(m_vertexData->vertexPositionBuffer[0]) * 3 * m_vertexData->vertexCount, 
		m_vertexData->vertexPositionBuffer, GL_STATIC_DRAW);

	glGenBuffers(1, &m_colorBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, m_colorBuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(m_vertexData->vertexColorBuffer[0]) * 3 * m_vertexData->vertexCount,
		m_vertexData->vertexColorBuffer, GL_STATIC_DRAW);

	return true;
}

void Mesh::Shutdown()
{
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
	

	glDeleteBuffers(1, &m_vertexBuffer);
	glDeleteVertexArrays(1, &m_vertexArrayID);
}

void Mesh::Draw()
{
	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);

	glBindBuffer(GL_ARRAY_BUFFER, m_vertexBuffer);
	glVertexAttribPointer(
		0,
		3,
		GL_FLOAT,
		GL_FALSE,
		0,
		(void*)0
		);

	glBindBuffer(GL_ARRAY_BUFFER, m_colorBuffer);
	glVertexAttribPointer(
		1,
		3,
		GL_FLOAT,
		GL_FALSE,
		0,
		(void*)0
		);
	
	glDrawArrays(GL_TRIANGLES, 0, m_vertexData->vertexCount);

	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);
	return;
}

void Mesh::Transform(const glm::vec3 position, const glm::vec3 rotation, const glm::vec3 scale_v)
{
	this->position = position;
	this->rotation = rotation;
	this->scale = scale;

	glm::mat4 translation, rotation_x, rotation_y, rotation_z, scale_m;
	translation = glm::translate(position);
	rotation_x = glm::rotate(rotation.x, glm::vec3(1.0f, 0.0f, 0.0f));
	rotation_y = glm::rotate(rotation.y, glm::vec3(0.0f, 1.0f, 0.0f));
	rotation_z = glm::rotate(rotation.z, glm::vec3(0.0f, 0.0f, 1.0f));
	scale_m = glm::scale(scale_v);

	modelMatrix = translation*(rotation_x*rotation_y*rotation_z)*scale_m;
}

glm::mat4* Mesh::GetModelMatrix()
{
	return &modelMatrix;
}

glm::vec3 Mesh::GetPosition()
{
	return position;
}

glm::vec3 Mesh::GetRotation()
{
	return rotation;
}

glm::vec3 Mesh::GetScale()
{
	return scale;
}
