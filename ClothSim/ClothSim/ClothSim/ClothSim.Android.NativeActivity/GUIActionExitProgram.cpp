#include "pch.h"
#include "GUIActionExitProgram.h"
#include "System.h"


GUIActionExitProgram::GUIActionExitProgram(GUIButton* b) : GUIAction(b)
{
}

GUIActionExitProgram::GUIActionExitProgram(const GUIActionExitProgram * c) : GUIAction(c)
{
}


GUIActionExitProgram::~GUIActionExitProgram()
{
}

unsigned int GUIActionExitProgram::Action()
{
	unsigned int err = CS_ERR_NONE;

	System::GetInstance()->Stop();

	return err;
}
