#pragma once
#include "GUIElement.h"
#include <vector>

class GUIActionMoveSliderHead;
class GUIActionSetSliderHead;
class GUIPicture;
class GUIButton;
class GUIText;

class GUISlider :
	public GUIElement
{
protected:
	std::vector<string> m_labelStrings;
	
	GUIText* m_textLabel;
	GUIText* m_textValue;
	GUIButton* m_sliderHead;
	GUIPicture* m_sliderBar;

	TextureID* m_headTex;
	TextureID* m_barTex;
	TextureID* m_fontTex;
	
	float m_length;
	unsigned int m_states;
	unsigned int m_defState;
	unsigned int m_currentState;

public:
	GUISlider(const std::string* name, TextureID* headTex, TextureID* barTex, TextureID* fontTex, unsigned int states, unsigned int defState);
	GUISlider(const GUISlider* c);
	~GUISlider();

	virtual unsigned int Initialize();
	virtual unsigned int Shutdown();
};

