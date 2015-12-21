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

unsigned int GUIActionExitProgram::Action(std::vector<void*>* params, const glm::vec2* clickPos)
{
	unsigned int err = CS_ERR_NONE;

	LOGI("Shutting down!");

	System::GetInstance()->Stop();

	return err;
}
