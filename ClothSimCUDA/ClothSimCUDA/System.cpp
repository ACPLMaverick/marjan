#include "System.h"

System* System::instance;

System::System()
{

}

System::System(const System*)
{

}


System::~System()
{

}

System* System::GetInstance()
{
	if (System::instance == nullptr)
		System::instance = new System();

	return System::instance;
}

void System::DestroyInstance()
{
	if (System::instance != nullptr)
		delete System::instance;
}

unsigned int System::Initialize()
{
	unsigned int err = CS_ERR_NONE;

	m_running = true;

	// initializing main singletons

	err = Renderer::GetInstance()->Initialize();
	if (err != CS_ERR_NONE) return err;

	err = InputManager::GetInstance()->Initialize();
	if (err != CS_ERR_NONE) return err;

	InputHandler::GetInstance();

	// initializing scene and objects

	// initializing gui

	// initializing CUDA

#ifdef _DEBUG
	printf("\nSystem.Initialize has completed successfully.\n");
#endif
	return err;
}

unsigned int System::Shutdown()
{
	unsigned int err = CS_ERR_NONE;

	m_running = false;

	// shutting down main singletons

	err = Renderer::GetInstance()->Shutdown();
	if (err != CS_ERR_NONE) return err;

	err = InputManager::GetInstance()->Shutdown();
	if (err != CS_ERR_NONE) return err;

	Renderer::DestroyInstance();
	InputManager::DestroyInstance();
	InputHandler::DestroyInstance();

	// shutting down scene and objects

	// shutting down gui

	// shutting down CUDA

#ifdef _DEBUG
	printf("\nSystem.Shutdown has completed successfully.\n");
#endif
	return err;
}

// main loop runs here
unsigned int System::Run()
{
	unsigned int err = CS_ERR_NONE;

	while (m_running)
	{
		// update input

		// update camera & gui

		// compute cloth physics

		// compute collisions

		// draw one frame
		Renderer::GetInstance()->Run();
	}

	return err;
}

