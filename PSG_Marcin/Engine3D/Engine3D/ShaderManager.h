#pragma once
#include <map>
#include <D3D11.h>
#include "TextureShader.h"
#include "LightShader.h"
#include "SpecularShader.h"
#include "DeferredShader.h"
#include "LightDeferredShader.h"

class ShaderManager
{
private:
	map<LPCSTR, TextureShader*> shaders;
public:
	ShaderManager();
	~ShaderManager();

	TextureShader* LoadShader(ID3D11Device* device, HWND hwnd, int id);
	bool AddShaders(ID3D11Device* device, HWND hwnd);
	void Shutdown();
};

