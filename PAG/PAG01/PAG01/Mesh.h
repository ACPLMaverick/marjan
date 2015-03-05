#pragma once
#include <GL\glew.h>
#include <GLFW\glfw3.h>
#include <glm\glm\glm.hpp>
#include <glm\glm\gtx\transform.hpp>

struct VertexData
{
	GLfloat* vertexPositionBuffer;
	GLfloat* vertexColorBuffer;
	GLfloat* vertexUVBuffer;
	GLfloat* vertexNormalBuffer;
	unsigned int vertexCount;
};

class Mesh
{
private:
	glm::mat4 modelMatrix;
	glm::vec3 position, rotation, scale;

	GLuint m_vertexArrayID;
	VertexData* m_vertexData;
	GLuint m_vertexBuffer;
	GLuint m_colorBuffer;

public:

	Mesh();
	~Mesh();

	bool Initialize();
	void Shutdown();

	void Draw();

	void Transform(const glm::vec3 position, const glm::vec3 rotation, const glm::vec3 scale);
	glm::vec3 GetPosition();
	glm::vec3 GetRotation();
	glm::vec3 GetScale();
	glm::mat4* GetModelMatrix();
};

