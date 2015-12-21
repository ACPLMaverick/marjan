#include "pch.h"
#include "GUISlider.h"


GUISlider::GUISlider(const std::string * name, TextureID * headTex, TextureID * barTex, TextureID * fontTex, unsigned int states, unsigned int defState) : GUIElement(name)
{
	m_headTex = headTex;
	m_barTex = barTex;
	m_fontTex = fontTex;
	m_states = states;
	m_defState = defState;
}

GUISlider::GUISlider(const GUISlider * c) : GUIElement(c)
{
}

GUISlider::~GUISlider()
{
}

unsigned int GUISlider::Initialize()
{
	unsigned int err = CS_ERR_NONE;

	err = GUIElement::Initialize();
	if (err != CS_ERR_NONE)
		return err;

	return err;
}

unsigned int GUISlider::Shutdown()
{
	unsigned int err = CS_ERR_NONE;

	err = GUIElement::Shutdown();
	if (err != CS_ERR_NONE)
		return err;

	return err;
}
