#include "pch.h"
#include "GUIAction.h"


GUIAction::GUIAction(GUIButton* b)
{
	m_button = b;
}

GUIAction::GUIAction(const GUIAction * c)
{
	m_button = c->m_button;
}

GUIAction::~GUIAction()
{
}

GUIButton * GUIAction::GetMyButton()
{
	return m_button;
}
