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

unsigned int GUIActionSetDisplayMode::Action(std::vector<void*>* params)
{
	unsigned int err = CS_ERR_NONE;

	LOGI("Changing diplay mode!");
	DrawMode m = Renderer::GetInstance()->GetDrawMode();
	int newMode = (((int)m + 1) % 3);
	Renderer::GetInstance()->SetDrawMode((DrawMode)newMode);

	return err;
}
