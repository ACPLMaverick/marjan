#pragma once
#include "GUIAction.h"
class GUIActionShowPreferences :
	public GUIAction
{
public:
	GUIActionShowPreferences(GUIButton* b);
	GUIActionShowPreferences(const GUIActionShowPreferences* c);
	~GUIActionShowPreferences();

	virtual unsigned int Action(void* params);
};

