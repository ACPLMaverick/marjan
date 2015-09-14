#pragma once

/*
	This is a mesh used to represent 2D text to draw on the screen.
*/

#define CHAR_HEIGHT 32
#define CHAR_WIDTH 32
#define FIELD_WIDTH 10
#define FIELD_HEIGHT 20
#define START_CHAR 32
#define CHAR_COUNT 256
#define SIZE_MULTIPLIER 0.1f

#include "MeshGL.h"

#include <string>

class MeshGLText :
	public MeshGL
{
protected:
	const string* m_text;
	unsigned long m_textLetterCount;

	virtual void GenerateVertexData();
	void UpdateVertexDataUV();
public:
	MeshGLText(SimObject*, const string*);
	MeshGLText(const MeshGLText*);
	~MeshGLText();

	virtual unsigned int Draw();

	const string* GetText();
	void SetText(const string* text);
};

