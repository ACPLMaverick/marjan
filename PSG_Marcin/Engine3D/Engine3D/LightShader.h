#pragma once

//includes
#include <d3d11.h>
#include <D3DX10math.h>
#include <D3DX11async.h>
#include <fstream>
#include "TextureShader.h"
#include "Globals.h"
using namespace std;

class LightShader : public TextureShader
{
protected:
	struct LightBuffer
	{
		D3DXVECTOR4 diffuseColor[LIGHT_MAX_COUNT];
		D3DXVECTOR4 lightDirection[LIGHT_MAX_COUNT];
	};

	struct AmbientBuffer
	{
		D3DXVECTOR4 ambientColor;
	};

	ID3D11Buffer* m_lightBuffer;
	ID3D11Buffer* m_ambientBuffer;

	virtual bool InitializeShader(ID3D11Device*, HWND, LPCSTR, LPCSTR);
	virtual void ShutdownShader();

	virtual bool SetShaderParameters(ID3D11DeviceContext*, D3DXMATRIX, D3DXMATRIX, D3DXMATRIX, ID3D11ShaderResourceView*, 
		D3DXVECTOR4 diffuseColors[], D3DXVECTOR4 lightDirections[], unsigned int lightCount, D3DXVECTOR4 ambientColor);
public:
	LightShader();
	LightShader(const LightShader&);
	~LightShader();

	virtual bool Initialize(ID3D11Device*, HWND, int);
	virtual bool Render(ID3D11DeviceContext*, int, D3DXMATRIX, D3DXMATRIX, D3DXMATRIX, ID3D11ShaderResourceView*,
		D3DXVECTOR4 diffuseColors[], D3DXVECTOR4 lightDirections[], unsigned int lightCount, D3DXVECTOR4 ambientColor);
};

