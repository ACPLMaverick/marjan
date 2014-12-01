#pragma once

//includes
#include <d3d11.h>
#include <d3dX11tex.h>
#include <iostream>

class Texture
{
private:
	ID3D11ShaderResourceView* m_texture;
public:
	std::string myName; 

	Texture();
	Texture(const Texture&);
	~Texture();

	bool Initialize(ID3D11Device*, LPCSTR);
	void Shutdown();

	ID3D11ShaderResourceView* GetTexture();
};

