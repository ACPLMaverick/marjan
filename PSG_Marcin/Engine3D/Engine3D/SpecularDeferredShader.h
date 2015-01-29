#pragma once
#include "SpecularShader.h"
class SpecularDeferredShader :
	public SpecularShader
{
protected:
	virtual bool InitializeShader(ID3D11Device*, HWND, LPCSTR, LPCSTR);
	virtual bool SetShaderParameters(ID3D11DeviceContext*, D3DXMATRIX, D3DXMATRIX, D3DXMATRIX, ID3D11ShaderResourceView*, ID3D11ShaderResourceView*, ID3D11ShaderResourceView*,
		D3DXVECTOR4 diffuseColors[], D3DXVECTOR4 lightDirections[], unsigned int lightCount, D3DXVECTOR4 ambientColor, D3DXVECTOR3 viewVector, D3DXVECTOR4 specularColor, float specularIntensity, float specularGlossiness);
public:
	SpecularDeferredShader();
	SpecularDeferredShader(const SpecularDeferredShader &other);
	~SpecularDeferredShader();

	virtual bool Initialize(ID3D11Device*, HWND, int);
	virtual bool Render(ID3D11DeviceContext*, int, D3DXMATRIX, D3DXMATRIX, D3DXMATRIX, ID3D11ShaderResourceView*, ID3D11ShaderResourceView*, ID3D11ShaderResourceView*,
		D3DXVECTOR4 diffuseColors[], D3DXVECTOR4 lightDirections[], unsigned int lightCount, D3DXVECTOR4 ambientColor, D3DXVECTOR3 viewVector, D3DXVECTOR4 specularColor, float specularIntensity, float specularGlossiness);
};

