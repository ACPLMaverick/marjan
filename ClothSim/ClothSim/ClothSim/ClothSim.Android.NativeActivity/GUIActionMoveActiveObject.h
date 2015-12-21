#pragma once
#include "GUIAction.h"
class GUIActionMoveActiveObject :
	public GUIAction
{
protected:
	enum MovementDirection
	{
		FORWARD = 1,
		BACKWARD,
		LEFT,
		RIGHT,
		UP,
		DOWN
	};

	const float BOX_SPEED = 0.005f;

public:
	GUIActionMoveActiveObject(GUIButton* b);
	GUIActionMoveActiveObject(const GUIActionMoveActiveObject* c);
	~GUIActionMoveActiveObject();

	virtual unsigned int Action(std::vector<void*>* params, const glm::vec2* clickPos);
};

