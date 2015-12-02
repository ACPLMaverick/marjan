#include "pch.h"
#include "GUIActionSetDisplayMode.h"


GUIActionSetDisplayMode::GUIActionSetDisplayMode(GUIButton* b) : GUIAction(b)
{
}

GUIActionSetDisplayMode::GUIActionSetDisplayMode(const GUIActionSetDisplayMode * c) : GUIAction(c)
{
}


GUIActionSetDisplayMode::~GUIActionSetDisplayMode()
{
}

unsigned int GUIActionSetDisplayMode::Action()
{
	unsigned int err = CS_ERR_NONE;



	return err;
}
