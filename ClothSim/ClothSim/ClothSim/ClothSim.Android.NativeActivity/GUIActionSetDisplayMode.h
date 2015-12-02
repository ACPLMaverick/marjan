#pragma once
#include "GUIAction.h"
class GUIActionSetDisplayMode :
	public GUIAction
{
public:
	GUIActionSetDisplayMode(GUIButton* b);
	GUIActionSetDisplayMode(const GUIActionSetDisplayMode* c);
	~GUIActionSetDisplayMode();

	virtual unsigned int Action();
};

