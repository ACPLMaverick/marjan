#pragma once
#include "GUIAction.h"
class GUIActionExitProgram :
	public GUIAction
{
public:
	GUIActionExitProgram(GUIButton* b);
	GUIActionExitProgram(const GUIActionExitProgram* c);
	~GUIActionExitProgram();

	virtual unsigned int Action(std::vector<void*>* params);
};

