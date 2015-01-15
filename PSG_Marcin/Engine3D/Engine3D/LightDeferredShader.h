#pragma once
#include "LightShader.h"
class LightDeferredShader :
	public LightShader
{
protected:
	virtual bool InitializeShader(ID3D11Device*, HWND, LPCSTR, LPCSTR);
	virtual bool SetShaderParameters(ID3D11DeviceContext*, D3DXMATRIX, D3DXMATRIX, D3DXMATRIX, ID3D11ShaderResourceView*, ID3D11ShaderResourceView*,
		D3DXVECTOR4 diffuseColors[], D3DXVECTOR4 lightDirections[], unsigned int lightCount, D3DXVECTOR4 ambientColor);
public:
	LightDeferredShader();
	LightDeferredShader(const LightDeferredShader &other);
	~LightDeferredShader();

	virtual bool Initialize(ID3D11Device*, HWND, int);
	virtual bool Render(ID3D11DeviceContext*, int, D3DXMATRIX, D3DXMATRIX, D3DXMATRIX, ID3D11ShaderResourceView*, ID3D11ShaderResourceView*,
		D3DXVECTOR4 diffuseColors[], D3DXVECTOR4 lightDirections[], unsigned int lightCount, D3DXVECTOR4 ambientColor);
};

