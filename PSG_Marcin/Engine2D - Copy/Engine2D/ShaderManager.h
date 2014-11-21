#pragma once
#include <map>
#include <D3D11.h>
#include "TextureShader.h"

class ShaderManager
{
private:
	map<LPCSTR, TextureShader*> shaders;
public:
	ShaderManager();
	~ShaderManager();

	TextureShader* LoadShader(ID3D11Device* device, HWND hwnd, LPCSTR path);
	TextureShader* LoadShader(ID3D11Device* device, HWND hwnd, int id);
	bool AddShader(ID3D11Device* device, HWND hwnd, LPCSTR path);
	void Shutdown();
};

