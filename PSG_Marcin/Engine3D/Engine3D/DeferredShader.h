#pragma once
#include "TextureShader.h"
class DeferredShader :
	public TextureShader
{
private:
	virtual bool InitializeShader(ID3D11Device*, HWND, LPCSTR, LPCSTR);
public:
	DeferredShader();
	DeferredShader(DeferredShader &other);
	~DeferredShader();

	virtual bool Initialize(ID3D11Device*, HWND, int);
};

