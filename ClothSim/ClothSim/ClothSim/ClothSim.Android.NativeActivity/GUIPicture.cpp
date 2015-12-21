#include "pch.h"
#include "GUIPicture.h"


GUIPicture::GUIPicture(const std::string* s, TextureID* tex) : GUIElement(s)
{
	m_textureIdle = tex;
}

GUIPicture::GUIPicture(const GUIPicture * c) : GUIElement(c)
{
}


GUIPicture::~GUIPicture()
{
}

unsigned int GUIPicture::Initialize()
{
	unsigned int err = CS_ERR_NONE;

	err = GUIElement::Initialize();
	if (err != CS_ERR_NONE)
		return err;

	err = m_mesh->Initialize();
	m_mesh->SetTextureID(m_textureIdle);

	return err;
}
