#include "pch.h"
#include "GUIActionShowPreferences.h"


GUIActionShowPreferences::GUIActionShowPreferences(GUIButton* b) : GUIAction(b)
{
}

GUIActionShowPreferences::GUIActionShowPreferences(const GUIActionShowPreferences * c) : GUIAction(c)
{
}


GUIActionShowPreferences::~GUIActionShowPreferences()
{
}

unsigned int GUIActionShowPreferences::Action(void* params)
{
	unsigned int err = CS_ERR_NONE;

	LOGI("Preferences!");

	return err;
}
