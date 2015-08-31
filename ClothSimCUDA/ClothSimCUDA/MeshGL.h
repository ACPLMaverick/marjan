#pragma once

/*
This is an abstract representation of a Mesh rendered by OpenGL 3.3.
*/

#include "Mesh.h"
#include "Renderer.h"

#include <GL\glew.h>
#include <GLFW\glfw3.h>
#include <glm\glm\glm.hpp>
#include <glm\glm\gtx\transform.hpp>

////////////

struct Vertex
{
	unsigned int id;
	glm::vec3* position;
	glm::vec2* uv;
	glm::vec3* normal;
	glm::vec4* color;
};

struct VertexDataRaw
{
	glm::vec3* positionBuffer;
	glm::vec3* uvBuffer;
	glm::vec3* normalBuffer;
	glm::vec4* colorBuffer;
	(unsigned int)* indexBuffer;
	unsigned int vertexCount;
	unsigned int indexCount;

	VertexDataRaw()
	{
		positionBuffer = nullptr;
		uvBuffer = nullptr;
		normalBuffer = nullptr;
		colorBuffer = nullptr;
		indexBuffer = nullptr;
		vertexCount = 0;
		indexCount = 0;
	}

	~VertexDataRaw()
	{
		if (positionBuffer != nullptr)
		{
			delete[] positionBuffer;
		}
		if (uvBuffer != nullptr)
		{
			delete[] uvBuffer;
		}
		if (normalBuffer != nullptr)
		{
			delete[] normalBuffer;
		}
		if (colorBuffer != nullptr)
		{
			delete[] colorBuffer;
		}
		if (indexBuffer != nullptr)
		{
			delete[] indexBuffer;
		}
	}
};

struct VertexDataID
{
	GLuint vertexArrayID;
	GLuint vertexBuffer;
	GLuint uvBuffer;
	GLuint normalBuffer;
	GLuint colorBuffer;
	GLuint indexBuffer;
};

struct VertexData
{
	VertexDataRaw* data;
	VertexDataID* ids;

	VertexData()
	{
		data = nullptr;
		ids = nullptr;
	}

	~VertexData()
	{
		if (data != nullptr)
		{
			delete data;
		}
		if (ids != nullptr)
		{
			delete ids;
		}
	}
};

////////////

class MeshGL :
	public Mesh
{
protected:
	VertexData* m_vertexData;
	Vertex* m_vertexArray;

	// ids for shader
	GLuint id_worldViewProj;
	GLuint id_world;
	GLuint id_worldInvTrans;
	GLuint id_eyeVector;
	GLuint id_lightDir;
	GLuint id_lightDiff;
	GLuint id_lightSpec;
	GLuint id_lightAmb;
	GLuint id_gloss;
	////

	virtual void GenerateVertexData() = 0;
public:
	MeshGL(SimObject*);
	MeshGL(const MeshGL*);
	~MeshGL();

	virtual unsigned int Initialize();
	virtual unsigned int Shutdown();

	virtual unsigned int Draw();

	virtual void UpdateShaderIDs(unsigned int);
};

