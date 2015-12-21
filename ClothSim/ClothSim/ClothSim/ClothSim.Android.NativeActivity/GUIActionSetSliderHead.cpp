#include "pch.h"
#include "GUIActionSetSliderHead.h"


GUIActionSetSliderHead::GUIActionSetSliderHead(GUIButton* b, float length, unsigned int options, unsigned int defOption) : GUIAction(b)
{
	m_length = length;
	m_options = options;
	m_currentOption = defOption;
}

GUIActionSetSliderHead::GUIActionSetSliderHead(const GUIActionSetSliderHead * c) : GUIAction(c)
{
}


GUIActionSetSliderHead::~GUIActionSetSliderHead()
{
}

unsigned int GUIActionSetSliderHead::Initialize()
{
	m_startPoint = m_button->GetPosition();
	m_startPoint.x -= m_length / 2.0f;
	m_step = m_length / (float)(m_options - 1);
	return CS_ERR_NONE;
}

unsigned int GUIActionSetSliderHead::Action(std::vector<void*>* params, const glm::vec2* clickPos)
{
	glm::vec2 cPos = m_button->GetPosition();
	float currX = cPos.x - m_startPoint.x;
	float sub = m_length + m_step;
	while ((sub -= m_step) / currX > 1.0f && sub > 0.0f) ;

	if (sub > 0.0f)
	{
		float diff1 = (sub + m_step) - currX;
		float diff2 = currX - sub;
		if (diff1 < diff2)	// go left
		{
			cPos.x = m_startPoint.x + sub;
		}
		else // go right
		{
			cPos.x = m_startPoint.x + sub + m_step;
		}
	}
	else
	{
		cPos.x = m_startPoint.x;
	}

	m_button->SetPosition(cPos);
	
	return CS_ERR_NONE;
}

unsigned int GUIActionSetSliderHead::GetCurrentOption()
{
	return m_currentOption;
}
