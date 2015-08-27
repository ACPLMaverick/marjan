#include "InputManager.h"
InputManager* InputManager::instance;

InputManager::InputManager()
{

}

InputManager::InputManager(const InputManager*)
{

}

InputManager::~InputManager()
{

}

InputManager* InputManager::GetInstance()
{
	if (InputManager::instance == nullptr)
	{
		InputManager::instance = new InputManager();
	}

	return InputManager::instance;
}

void InputManager::DestroyInstance()
{
	if (InputManager::instance != nullptr)
		delete InputManager::instance;
}

unsigned int InputManager::Initialize()
{
	unsigned int err = CS_ERR_NONE;

	return err;
}

unsigned int InputManager::Shutdown()
{
	unsigned int err = CS_ERR_NONE;

	return err;
}

unsigned int InputManager::Run()
{
	unsigned int err = CS_ERR_NONE;

	return err;
}
