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
	(unsigned int)* indexBuffer;
	unsigned int vertexCount;
	unsigned int indexCount;
};

struct VertexID
{
	GLuint vertexArrayID;
	GLuint vertexBuffer;
	GLuint colorBuffer;
	GLuint uvBuffer;
	GLuint normalBuffer;
	GLuint indexBuffer;
};

struct BoundingSphere
{
	glm::vec4 d_position;
	GLfloat d_radius;
	glm::vec4 position;
	GLfloat radius;
};

class Mesh
{
private:
	string m_name;

	glm::mat4 modelMatrix, mvpMatrix, rotationOnlyMatrix, translationOnlyMatrix, scaleOnlyMatrix;
	glm::vec3 position, rotation, scale;
	GLuint highlightID;
	glm::vec4 highlight;

	// hierarchy
	Mesh* parent;
	vector<Mesh*> children;
	////

	BoundingSphere* boundingSphere;

	VertexData* m_vertexData;
	VertexID* m_vertexID;

	Texture* m_texture;

public:
	GLuint mvpMatrixID;
	GLuint modelID;
	short myID, parentID;
	bool visible;

	Mesh();
	~Mesh();

	bool Initialize(GLuint programID, Mesh* parent, string name, VertexData* data, BoundingSphere* bs, short myID, short parentID);
	void Shutdown();

	void Draw(glm::mat4* projectionMatrix, glm::mat4* viewMatrix, glm::vec3* eyeVector, GLuint eyeVectorID, Light* light);

	void Transform(const glm::vec3* position, const glm::vec3* rotation, const glm::vec3* scale);
	void Transform(const glm::mat4* matrix);
	glm::vec3* GetPosition();
	glm::vec3* GetRotation();
	glm::vec3* GetScale();
	glm::mat4* GetModelMatrix();
	glm::mat4* GetRotationOnlyMatrix();
	glm::mat4* GetTranslationOnlyMatrix();
	glm::mat4* GetScaleOnlyMatrix();
	void SetTexture(Texture* texture);

	void AddChild(Mesh* child);
	vector<Mesh*>* GetChildren();
	Mesh* GetParent();
	void SetParent(Mesh* parent);
	BoundingSphere* GetBoundingSphere();

	void Highlight();
	void DisableHighlight();
};

