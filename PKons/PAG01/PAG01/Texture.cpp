#include "Texture.h"


Texture::Texture()
{
	m_texWidth = 0;
	m_texHeight = 0;
}


Texture::~Texture()
{
}

bool Texture::Initialize(const string* filePath)
{
	LoadFromFile(filePath);
	if (m_ID == 0) return false;

	return true;
}

void Texture::Shutdown()
{

}

void Texture::LoadFromFile(const string* filePath)
{
	m_ID = SOIL_load_OGL_texture((*filePath).c_str(), SOIL_LOAD_AUTO, SOIL_CREATE_NEW_ID,
		SOIL_FLAG_MIPMAPS | SOIL_FLAG_INVERT_Y | SOIL_FLAG_NTSC_SAFE_RGB | SOIL_FLAG_COMPRESS_TO_DXT);
}

GLuint Texture::GetID()
{
	return m_ID;
}
