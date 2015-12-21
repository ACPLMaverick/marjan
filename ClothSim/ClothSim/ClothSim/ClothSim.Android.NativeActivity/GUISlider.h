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
	std::vector<string> m_valStrings;
	string m_label;
	
	GUIText* m_textLabel;
	GUIText* m_textValue;
	GUIButton* m_sliderHead;
	GUIPicture* m_sliderBar;

	GUIActionMoveSliderHead* m_actionMove;
	GUIActionSetSliderHead* m_actionSet;

	TextureID* m_headTex;
	TextureID* m_barTex;
	TextureID* m_fontTex;
	
	float m_length;
	float m_labelMultiplier;
	float m_labelOffset;
	unsigned int m_states;
	unsigned int m_defState;
	unsigned int m_currentState;

	bool labelInitialized = false;

public:
	std::vector<std::function<void(unsigned int)>> EventStateChanged;

	GUISlider(const std::string* name, const std::string* label, TextureID* headTex, TextureID* barTex, TextureID* fontTex, unsigned int states, unsigned int defState, float labelMultiplier, float labelOffset);
	GUISlider(const std::string* name, const std::string* label, TextureID* headTex, TextureID* barTex, TextureID* fontTex, unsigned int states, unsigned int defState, std::vector<string>* labels);
	GUISlider(const GUISlider* c);
	~GUISlider();

	virtual unsigned int Initialize();
	virtual unsigned int Shutdown();

	virtual unsigned int Update();
};

