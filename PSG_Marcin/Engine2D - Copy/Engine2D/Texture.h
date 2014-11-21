#pragma once

//includes
#include <d3d11.h>
#include <d3dX11tex.h>

class Texture
{
private:
	ID3D11ShaderResourceView* m_texture;
public:
	Texture();
	Texture(const Texture&);
	~Texture();

	bool Initialize(ID3D11Device*, LPCSTR);
	void Shutdown();

	ID3D11ShaderResourceView* GetTexture();
};

