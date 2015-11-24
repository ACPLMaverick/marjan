#pragma once

/*
*/

#include "Common.h"
#include "Singleton.h"
#include "InputManager.h"

#include <glm\glm\glm.hpp>

class InputHandler : public Singleton<InputHandler>
{
	friend class Singleton<InputHandler>;

protected:
	InputHandler();
public:
	InputHandler(const InputHandler*);
	~InputHandler();

	bool ExitPressed();
	void GetCursorPosition(glm::vec2* vec);
	void GetCursorVector(glm::vec2* vec);
	void GetArrowsMovementVector(glm::vec3* vec);
	int GetZoomValue();
	bool ActionButtonPressed();
	bool ActionButtonClicked();
	bool CameraRotateButtonPressed();
	bool CameraMoveButtonPressed();
	bool WireframeButtonClicked();
};

