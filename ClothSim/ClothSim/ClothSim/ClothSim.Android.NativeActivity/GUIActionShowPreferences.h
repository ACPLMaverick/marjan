#pragma once
#include "GUIAction.h"
class GUIActionShowPreferences :
	public GUIAction
{
protected:
	const string MS_VALUE = "Mass-spring";
	const string PB_VALUE = "Position based";
	const string UN_VALUE = "Unknown";
public:
	GUIActionShowPreferences(GUIButton* b);
	GUIActionShowPreferences(const GUIActionShowPreferences* c);
	~GUIActionShowPreferences();

	virtual unsigned int Initialize();
	virtual unsigned int Action(std::vector<void*>* params);
};

