#include "System.h"
System* System::me;

System::System()
{
	m_graphics = nullptr;
	m_input = nullptr;
}


System::~System()
{
}

System* System::GetInstance()
{
	if (me == NULL)
	{
		System::me = new System();
	}
	return System::me;
}

void System::DestroyInstance()
{
	if (System::me != NULL)
	{
		delete System::me;
	}
}

void System::Initialize()
{
	m_graphics = new Graphics();
	m_graphics->Initialize();

	m_input = new Input();
	m_input->Initialize(m_graphics->GetWindowPtr(), MouseScrollCallback);

	isRunning = true;

	glfwSetCursorPos(m_graphics->GetWindowPtr(), WINDOW_WIDTH / 2, WINDOW_HEIGHT / 2);
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
		ProcessInput();

		// Render one frame.
		m_graphics->Frame();
	}
}

void System::ProcessInput()
{
	// Check if we should quit
	if (m_input->IsKeyDown(GLFW_KEY_ESCAPE)) isRunning = false;

	// process mouse to rotate camera
	if (m_input->IsMouseButtonDown(GLFW_MOUSE_BUTTON_2))
	{
		glfwGetCursorPos(m_graphics->GetWindowPtr(), &m_input->mouseX, &m_input->mouseY);

		m_input->horizontalAngle = MOUSE_SPEED * (WINDOW_WIDTH / 2 - m_input->mouseX);

		m_input->verticalAngle = MOUSE_SPEED * (WINDOW_HEIGHT / 2 - m_input->mouseY);

		m_input->verticalActual += m_input->verticalAngle;
		if (m_input->verticalActual > (1.4f) || m_input->verticalActual < (-1.4f))
		{
			m_input->verticalActual -= m_input->verticalAngle;
			m_input->verticalAngle = 0.0f;
		}

		Camera* cam = m_graphics->GetCameraPtr();
		mat4 rotationPitch, rotationYaw;
		rotationYaw = rotate(m_input->horizontalAngle, cam->GetUp());
		rotationPitch = rotate(m_input->verticalAngle, cam->GetRight());
		vec4 newPosition, /*newUp,*/ newRight;
		newPosition = vec4(cam->GetPosition(), 1.0f);
		//newUp = vec4(cam->GetUp(), 1.0f);
		newRight = vec4(cam->GetRight(), 1.0f);

		newPosition = newPosition*rotationPitch*rotationYaw;
		//newUp = newUp*rotationPitch;
		newRight = newRight*rotationYaw;

		m_graphics->GetCameraPtr()->Transform(
			&vec3(newPosition),
			&vec3(0.0f, 0.0f, 0.0f),
			&cam->GetUp(),
			&vec3(newRight)
			);
	}

	// reset mouse for the next frame
	glfwSetCursorPos(m_graphics->GetWindowPtr(), WINDOW_WIDTH / 2, WINDOW_HEIGHT / 2);
}

void System::MouseScrollCallback(GLFWwindow* window, double x, double y)
{
	//printf((to_string(x) + " " + to_string(y) + " " + to_string(System::me->m_graphics->GetCameraPtr()->GetPositionLength()) + "\n").c_str());

	if (System::me != NULL)
	{
		int direction = (int)y;
		if ((System::me->m_graphics->GetCameraPtr()->GetPositionLength()) + (float)direction*SCROLL_SPEED >= 0.01f)
		{
			Graphics* graph = System::me->m_graphics;
			Camera* cam = graph->GetCameraPtr();
			cam->Transform(&cam->GetPosition(), &cam->GetTarget(), &cam->GetUp(), &cam->GetRight(),
				cam->GetPositionLength() + (float)direction*SCROLL_SPEED);
		}
	}
}
