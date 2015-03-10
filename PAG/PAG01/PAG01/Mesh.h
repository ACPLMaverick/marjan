#pragma once
#include <GL\glew.h>
#include <GLFW\glfw3.h>
#include <glm\glm\glm.hpp>
#include <glm\glm\gtx\transform.hpp>
#include <vector>

#include "Texture.h"
#include "Light.h"

struct VertexData
{
	GLfloat* vertexPositionBuffer;
	GLfloat* vertexColorBuffer;
	GLfloat* vertexUVBuffer;
	GLfloat* vertexNormalBuffer;
	unsigned int vertexCount;
};

struct VertexID
{
	GLuint vertexArrayID;
	GLuint vertexBuffer;
	GLuint colorBuffer;
	GLuint uvBuffer;
	GLuint normalBuffer;
};

class Mesh
{
private:
	glm::mat4 modelMatrix, mvpMatrix;
	glm::vec3 position, rotation, scale;

	// hierarchy
	Mesh* parent;
	vector<Mesh*> children;
	////

	VertexData* m_vertexData;
	VertexID* m_vertexID;

	Texture* m_texture;

public:
	GLuint mvpMatrixID;

	Mesh();
	~Mesh();

	bool Initialize(GLuint programID, Mesh* parent);
	void Shutdown();

	void Draw(glm::mat4* projectionMatrix, glm::mat4* viewMatrix, glm::vec3* eyeVector, GLuint eyeVectorID, Light* light);

	void Transform(const glm::vec3* position, const glm::vec3* rotation, const glm::vec3* scale);
	glm::vec3* GetPosition();
	glm::vec3* GetRotation();
	glm::vec3* GetScale();
	glm::mat4* GetModelMatrix();
	void SetTexture(Texture* texture);

	void AddChild(Mesh* child);
	vector<Mesh*>* GetChildren();
	Mesh* GetParent();
	void SetParent(Mesh* parent);
};

