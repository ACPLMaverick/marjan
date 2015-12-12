#pragma once

/*
This is an abstract representation of a Mesh rendered by OpenGL 3.3.
*/

#include "Mesh.h"
#include "Renderer.h"
#include "LightAmbient.h"
#include "LightDirectional.h"

#include <EGL/egl.h>
#include <GLES/gl.h>
#include <glm\glm\glm.hpp>
#include <glm\glm\gtx\transform.hpp>

////////////

struct VertexDataRaw
{
	glm::vec4* positionBuffer;
	glm::vec2* uvBuffer;
	glm::vec4* normalBuffer;
	glm::vec4* colorBuffer;
	glm::vec4* barycentricBuffer;
	GLuint* indexBuffer;
	unsigned int vertexCount;
	unsigned int indexCount;

	VertexDataRaw()
	{
		positionBuffer = nullptr;
		uvBuffer = nullptr;
		normalBuffer = nullptr;
		colorBuffer = nullptr;
		indexBuffer = nullptr;
		barycentricBuffer = nullptr;
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
		if (barycentricBuffer != nullptr)
		{
			delete[] barycentricBuffer;
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
	GLuint barycentricBuffer;
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

	virtual void GenerateVertexData() = 0;
	void CreateVertexDataBuffers(unsigned int, unsigned int, GLenum);
	void GenerateBarycentricCoords();
public:
	MeshGL(SimObject*);
	MeshGL(const MeshGL*);
	~MeshGL();

	virtual unsigned int Initialize();
	virtual unsigned int Shutdown();

	virtual unsigned int Update();
	virtual unsigned int Draw();
};

