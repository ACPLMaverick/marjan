#include "pch.h"
#include "GUIActionMoveSliderHead.h"


GUIActionMoveSliderHead::GUIActionMoveSliderHead(GUIButton* b, float length) : GUIAction(b)
{
	m_length = length;
}

GUIActionMoveSliderHead::GUIActionMoveSliderHead(const GUIActionMoveSliderHead * c) : GUIAction(c)
{
}


GUIActionMoveSliderHead::~GUIActionMoveSliderHead()
{
}

unsigned int GUIActionMoveSliderHead::Initialize()
{
	m_startPoint = m_button->GetPosition();
	m_startPoint.x -= m_length / 2.0f;
	return CS_ERR_NONE;
}

unsigned int GUIActionMoveSliderHead::Action(std::vector<void*>* params, const glm::vec2* clickPos)
{
	if (clickPos->x >= m_startPoint.x && clickPos->x <= m_startPoint.x + m_length)
	{
		glm::vec2 cPos = m_button->GetPosition();
		glm::vec2 nPos = glm::vec2(clickPos->x, cPos.y);
		m_button->SetPosition(nPos);
	}
	return CS_ERR_NONE;
}
