#include "InputHandler.h"
InputHandler* InputHandler::instance;

InputHandler::InputHandler()
{

}

InputHandler::InputHandler(const InputHandler*)
{

}

InputHandler::~InputHandler()
{

}

InputHandler* InputHandler::GetInstance()
{
	if (InputHandler::instance == nullptr)
	{
		InputHandler::instance = new InputHandler();
	}

	return InputHandler::instance;
}

void InputHandler::DestroyInstance()
{
	if (InputHandler::instance != nullptr)
		delete InputHandler::instance;
}



bool InputHandler::ExitPressed()
{
	if (InputManager::GetInstance()->GetKey(GLFW_KEY_ESCAPE))
	{
		return true;
	}
	else return false;
}

glm::vec2 InputHandler::GetCursorPosition()
{
	glm::vec2 pos;
	MouseData* m = InputManager::GetInstance()->GetMouseData();

	pos.x = m->mouseX;
	pos.y = m->mouseY;

	return pos;
}

glm::vec2 InputHandler::GetCursorVector()
{
	glm::vec2 pos;
	MouseData* m = InputManager::GetInstance()->GetMouseData();

	pos.x = m->mouseXRelative;
	pos.y = m->mouseYRelative;

	return pos;
}

int InputHandler::GetZoomValue()
{
	MouseData* m = InputManager::GetInstance()->GetMouseData();
	return m->scrollRelative;
}

bool InputHandler::ActionButtonPressed()
{
	return InputManager::GetInstance()->GetKey(GLFW_MOUSE_BUTTON_LEFT);
}

bool InputHandler::ActionButtonClicked()
{
	return false;
}

bool InputHandler::CameraRotateButtonPressed()
{
	return InputManager::GetInstance()->GetKey(GLFW_MOUSE_BUTTON_RIGHT);
}

bool InputHandler::CameraMoveButtonPressed()
{
	return InputManager::GetInstance()->GetKey(GLFW_MOUSE_BUTTON_MIDDLE);
}

bool InputHandler::WireframeButtonClicked()
{
	return InputManager::GetInstance()->GetKey(GLFW_KEY_W);
}
