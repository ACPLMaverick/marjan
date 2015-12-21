#pragma once
#include "GUIAction.h"
class GUIActionMoveSliderHead :
	public GUIAction
{
protected:
	float m_length;
	glm::vec2 m_startPoint;
public:
	GUIActionMoveSliderHead(GUIButton* b, float length);
	GUIActionMoveSliderHead(const GUIActionMoveSliderHead* c);
	~GUIActionMoveSliderHead();

	virtual unsigned int Initialize();
	virtual unsigned int Action(std::vector<void*>* params, const glm::vec2* clickPos);
};

