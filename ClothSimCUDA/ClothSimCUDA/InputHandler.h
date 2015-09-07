#pragma once

/*
*/

#include "Common.h"
#include "InputManager.h"

#include <glm\glm\glm.hpp>

class InputHandler
{
protected:
	static InputHandler* instance;
	InputHandler();
public:
	InputHandler(const InputHandler*);
	~InputHandler();

	static InputHandler* GetInstance();
	static void DestroyInstance();


	bool ExitPressed();
	glm::vec2 GetCursorPosition();
	glm::vec2 GetCursorVector();
	int GetZoomValue();
	bool ActionButtonPressed();
	bool ActionButtonClicked();
	bool CameraRotateButtonPressed();
	bool CameraMoveButtonPressed();
	bool WireframeButtonClicked();
};

