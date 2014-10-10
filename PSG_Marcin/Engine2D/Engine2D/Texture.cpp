#include "Texture.h"


Texture::Texture()
{
	m_texture = nullptr;
}

Texture::Texture(const Texture&)
{
}

Texture::~Texture()
{
}

bool Texture::Initialize(ID3D11Device*, LPCSTR)
{

}

void Texture::Shutdown()
{

}

ID3D11ShaderResourceView* Texture::GetTexture()
{

}
