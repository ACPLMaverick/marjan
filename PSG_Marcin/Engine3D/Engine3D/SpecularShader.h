#pragma once

//includes
#include <d3d11.h>
#include <D3DX10math.h>
#include <D3DX11async.h>
#include <fstream>
#include "LightShader.h"
using namespace std;

class SpecularShader : public LightShader
{
protected:
	struct SpecularBuffer
	{
		D3DXVECTOR3 viewVector;
		float specularIntensity;
		D3DXVECTOR4 specularColor;
		float glossiness;
		D3DXVECTOR3 padding02;
		D3DXVECTOR4 padding01;
	};

	ID3D11Buffer* m_specularBuffer;

	virtual bool InitializeShader(ID3D11Device*, HWND, LPCSTR, LPCSTR);
	virtual void ShutdownShader();

	virtual bool SetShaderParameters(ID3D11DeviceContext*, D3DXMATRIX, D3DXMATRIX, D3DXMATRIX, ID3D11ShaderResourceView*,
		D3DXVECTOR4 diffuseColors[], D3DXVECTOR4 lightDirections[], unsigned int lightCount, D3DXVECTOR4 ambientColor, D3DXVECTOR3 viewVector,
		D3DXVECTOR4 specularColor, float specularIntensity, float specularGlossiness);
public:
	SpecularShader();
	SpecularShader(const SpecularShader&);
	~SpecularShader();

	virtual bool Initialize(ID3D11Device*, HWND, int);
	virtual bool Render(ID3D11DeviceContext*, int, D3DXMATRIX, D3DXMATRIX, D3DXMATRIX, ID3D11ShaderResourceView*, 
		D3DXVECTOR4 diffuseColors[], D3DXVECTOR4 lightDirections[], unsigned int lightCount, D3DXVECTOR4 ambientColor, D3DXVECTOR3 viewVector, D3DXVECTOR4 specularColor, float specularIntensity, float specularGlossiness);
};

