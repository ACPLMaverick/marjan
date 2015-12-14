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

unsigned int GUIActionExitProgram::Action(std::vector<void*>* params)
{
	unsigned int err = CS_ERR_NONE;

	LOGI("Shitting down! (your neck)");

	System::GetInstance()->Stop();

	return err;
}
