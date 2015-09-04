#include "SimController.h"


SimController::SimController(SimObject* obj) : Component(obj)
{
}

SimController::SimController(const SimController* c) : Component(c)
{
}


SimController::~SimController()
{
}

unsigned int SimController::Initialize()
{
	return CS_ERR_NONE;
}

unsigned int SimController::Shutdown()
{
	return CS_ERR_NONE;
}



unsigned int SimController::Update()
{
	if (InputHandler::GetInstance()->ExitPressed())
	{
		System::GetInstance()->Stop();
	}

	return CS_ERR_NONE;
}

unsigned int SimController::Draw()
{
	return CS_ERR_NONE;
}