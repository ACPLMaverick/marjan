#pragma once
#include "MeshGLRect.h"
#include "GUIButton.h"

class GUIButton;

class MeshGLRectButton :
	public MeshGLRect
{
protected:
	GUIButton* m_btn;
	ShaderID* m_fontShaderID;

public:
	MeshGLRectButton(SimObject* obj, GUIButton* btn, glm::vec4* col);
	MeshGLRectButton(const MeshGLRectButton* c);
	~MeshGLRectButton();

	virtual unsigned int Initialize();
	virtual unsigned int Draw();
};

