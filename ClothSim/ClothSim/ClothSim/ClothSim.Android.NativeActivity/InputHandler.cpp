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



bool InputHandler::ExitPressed()
{
	if (InputManager::GetInstance()->GetTouch())
	{
		return true;
	}
	else return false;
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

bool InputHandler::ActionButtonPressed()
{
	return false;
}

bool InputHandler::ActionButtonClicked()
{
	return false;
}

bool InputHandler::CameraRotateButtonPressed()
{
	return InputManager::GetInstance()->GetTouch();
}

bool InputHandler::CameraMoveButtonPressed()
{
	return InputManager::GetInstance()->GetDoubleTouch();
}

bool InputHandler::WireframeButtonClicked()
{
	return InputManager::GetInstance()->GetTouch();
}
