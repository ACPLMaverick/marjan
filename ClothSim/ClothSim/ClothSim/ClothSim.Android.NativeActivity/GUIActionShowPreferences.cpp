#include "pch.h"
#include "GUIActionShowPreferences.h"
#include "ClothSimulator.h"

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

	/////// temporary

	ClothSimulator* cSim = (ClothSimulator*)params;
	cSim->SwitchMode();

	/////////////////

	return err;
}
