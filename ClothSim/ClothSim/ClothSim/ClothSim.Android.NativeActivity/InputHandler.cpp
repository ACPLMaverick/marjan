#include "InputHandler.h"

InputHandler::InputHandler()
{

}

InputHandler::InputHandler(const InputHandler*)
{

}

InputHandler::~InputHandler()
{

}

bool InputHandler::GetPressed()
{
	return InputManager::GetInstance()->GetPress();
}

bool InputHandler::GetHold()
{
	bool ret = false;
	if (!GetPressed())
	{
		ret = InputManager::GetInstance()->GetTouch();
	}
	return ret;
}


void InputHandler::GetCameraMovementVector(glm::vec2* vec)
{
	InputManager::GetInstance()->GetDoubleTouchDirection(vec);
}

void InputHandler::GetCameraRotationVector(glm::vec2* vec)
{
	InputManager::GetInstance()->GetTouchDirection(vec);
}

float InputHandler::GetZoomValue()
{
	return InputManager::GetInstance()->GetPinchValue();
}

bool InputHandler::GetZoom()
{
	return InputManager::GetInstance()->GetPinch();
}

void InputHandler::GetClickPosition(glm::vec2 * vec)
{
	InputManager::GetInstance()->GetTouchPosition(vec);
}
