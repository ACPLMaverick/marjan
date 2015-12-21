#pragma once
#include "GUIAction.h"
class GUIActionSetSliderHead :
	public GUIAction
{
protected:
	float m_length;
	float m_step;
	unsigned int m_options;
	unsigned int m_currentOption;
	glm::vec2 m_startPoint;
public:
	GUIActionSetSliderHead(GUIButton* b, float length, unsigned int options, unsigned int defOption);
	GUIActionSetSliderHead(const GUIActionSetSliderHead* c);
	~GUIActionSetSliderHead();

	virtual unsigned int Initialize();
	virtual unsigned int Action(std::vector<void*>* params, const glm::vec2* clickPos);

	unsigned int GetCurrentOption();
};

