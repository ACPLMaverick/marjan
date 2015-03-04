#include "System.h"


System::System()
{
	m_graphics = nullptr;
	m_input = nullptr;
}


System::~System()
{
}

void System::Initialize()
{
	m_graphics = new Graphics();
	m_graphics->Initialize();

	m_input = new Input();
	m_input->Initialize(m_graphics->GetWindowPtr());

	isRunning = true;
}

void System::Shutdown()
{
	if (m_graphics != nullptr)
	{
		m_graphics->Shutdown();
		delete m_graphics;
	}
	if (m_input != nullptr)
	{
		m_input->Shutdown();
		delete m_input;
	}
}

void System::GameLoop()
{
	while (isRunning)
	{
		// Check if we should quit
		if (m_input->IsKeyDown(GLFW_KEY_ESCAPE)) isRunning = false;

		// Render one frame.
		m_graphics->Frame();
	}
}
