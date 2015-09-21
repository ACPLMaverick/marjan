#include "MeshGLPlane.h"

MeshGLPlane::MeshGLPlane(SimObject* obj) : MeshGL(obj)
{
	m_width = 10.0f;
	m_length = 10.0f;
	m_edgesWidth = 10;
	m_edgesLength = 10;
}

MeshGLPlane::MeshGLPlane(SimObject* obj, float width, float length, unsigned int edWidth, unsigned int edLength) : MeshGL(obj)
{
	m_width = width;
	m_length = length;
	m_edgesWidth = edWidth;
	m_edgesLength = edLength;
}


MeshGLPlane::~MeshGLPlane()
{
}



void MeshGLPlane::GenerateVertexData()
{
	//CreateVertexDataBuffers(24, 36, GL_DYNAMIC_DRAW);
}