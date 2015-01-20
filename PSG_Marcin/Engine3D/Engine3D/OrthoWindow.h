#pragma once
#include "Model.h"
class OrthoWindow :
	public Model
{
protected:
	unsigned int mWindowWidth;
	unsigned int mWindowHeight;
	virtual VertexIndex* LoadGeometry(bool ind, string filePath);
public:
	OrthoWindow();
	OrthoWindow(const OrthoWindow &other);
	~OrthoWindow();

	virtual bool Initialize(ID3D11Device*, unsigned int windowWidth, unsigned int windowHeight);
};

