#pragma once

#include "MeshGL.h"

class MeshGLRect :
	public MeshGL
{
protected:
	virtual void GenerateVertexData();
public:
	MeshGLRect(SimObject*);
	MeshGLRect(const MeshGLRect*);
	~MeshGLRect();
};

