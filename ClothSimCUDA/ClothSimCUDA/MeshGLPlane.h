#pragma once

/*
	This class represents a plane of a varying mesh detail, which will be used as a cloth base.
	It also contains data structure that allow simulation to access each vertex's neighbours and modify them.
*/

#include "MeshGL.h"
class MeshGLPlane :
	public MeshGL
{
protected:
	float m_width;
	float m_length;
	unsigned int m_edgesWidth;
	unsigned int m_edgesLength;

	virtual void GenerateVertexData();
public:
	MeshGLPlane(SimObject*);
	MeshGLPlane(SimObject*, float, float, unsigned int, unsigned int);
	MeshGLPlane(const MeshGLPlane*);
	~MeshGLPlane();
};

