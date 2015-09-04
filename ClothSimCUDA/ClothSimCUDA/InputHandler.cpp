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