#include "System.h"
System* System::me;

System::System()
{
	m_graphics = nullptr;
	m_input = nullptr;
	m_timer = nullptr;
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

	m_timer = new Timer();
	m_timer->Initialize();

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
	if (m_timer != nullptr)
	{
		m_timer->Shutdown();
		delete m_timer;
	}
}

void System::GameLoop()
{
	while (isRunning)
	{
		m_timer->Update();

		ProcessInput();

		// Render one frame.
		m_graphics->Frame();
	}
}

void System::ProcessInput()
{
	// Check if we should quit
	if (m_input->IsKeyDown(GLFW_KEY_ESCAPE)) isRunning = false;

	if (m_input->IsKeyDown(GLFW_KEY_W))
	{
		m_graphics->GetCurrentlySelected()->Transform(
			&((*m_graphics->GetCurrentlySelected()->GetPosition()) + vec3(0.0f, 0.0f, TRANSLATION_AMOUNT*m_timer->GetDeltaTime())),
			m_graphics->GetCurrentlySelected()->GetRotation(),
			m_graphics->GetCurrentlySelected()->GetScale()
			);
	}
	if (m_input->IsKeyDown(GLFW_KEY_S))
	{
		m_graphics->GetCurrentlySelected()->Transform(
			&((*m_graphics->GetCurrentlySelected()->GetPosition()) + vec3(0.0f, 0.0f, -TRANSLATION_AMOUNT*m_timer->GetDeltaTime())),
			m_graphics->GetCurrentlySelected()->GetRotation(),
			m_graphics->GetCurrentlySelected()->GetScale()
			);
	}
	if (m_input->IsKeyDown(GLFW_KEY_A))
	{
		m_graphics->GetCurrentlySelected()->Transform(
			&((*m_graphics->GetCurrentlySelected()->GetPosition()) + vec3(TRANSLATION_AMOUNT*m_timer->GetDeltaTime(), 0.0f, 0.0f)),
			m_graphics->GetCurrentlySelected()->GetRotation(),
			m_graphics->GetCurrentlySelected()->GetScale()
			);
	}
	if (m_input->IsKeyDown(GLFW_KEY_D))
	{
		m_graphics->GetCurrentlySelected()->Transform(
			&((*m_graphics->GetCurrentlySelected()->GetPosition()) + vec3(-TRANSLATION_AMOUNT*m_timer->GetDeltaTime(), 0.0f, 0.0f)),
			m_graphics->GetCurrentlySelected()->GetRotation(),
			m_graphics->GetCurrentlySelected()->GetScale()
			);
	}
	if (m_input->IsKeyDown(GLFW_KEY_Q))
	{
		m_graphics->GetCurrentlySelected()->Transform(
			&((*m_graphics->GetCurrentlySelected()->GetPosition()) + vec3(0.0f, TRANSLATION_AMOUNT*m_timer->GetDeltaTime(), 0.0f)),
			m_graphics->GetCurrentlySelected()->GetRotation(),
			m_graphics->GetCurrentlySelected()->GetScale()
			);
	}
	if (m_input->IsKeyDown(GLFW_KEY_Z))
	{
		m_graphics->GetCurrentlySelected()->Transform(
			&((*m_graphics->GetCurrentlySelected()->GetPosition()) + vec3(0.0f, -TRANSLATION_AMOUNT*m_timer->GetDeltaTime(), 0.0f)),
			m_graphics->GetCurrentlySelected()->GetRotation(),
			m_graphics->GetCurrentlySelected()->GetScale()
			);
	}
	if (m_input->IsKeyDown(GLFW_KEY_E))
	{
		m_graphics->GetCurrentlySelected()->Transform(
			m_graphics->GetCurrentlySelected()->GetPosition(),
			&((*m_graphics->GetCurrentlySelected()->GetRotation()) + vec3(ROTATION_AMOUNT*m_timer->GetDeltaTime(), 0.0f, 0.0f)),
			m_graphics->GetCurrentlySelected()->GetScale()
			);
	}
	if (m_input->IsKeyDown(GLFW_KEY_R))
	{
		m_graphics->GetCurrentlySelected()->Transform(
			m_graphics->GetCurrentlySelected()->GetPosition(),
			&((*m_graphics->GetCurrentlySelected()->GetRotation()) + vec3(0.0f, ROTATION_AMOUNT*m_timer->GetDeltaTime(), 0.0f)),
			m_graphics->GetCurrentlySelected()->GetScale()
			);
	}
	if (m_input->IsKeyDown(GLFW_KEY_T))
	{
		m_graphics->GetCurrentlySelected()->Transform(
			m_graphics->GetCurrentlySelected()->GetPosition(),
			&((*m_graphics->GetCurrentlySelected()->GetRotation()) + vec3(0.0f, 0.0f, ROTATION_AMOUNT*m_timer->GetDeltaTime())),
			m_graphics->GetCurrentlySelected()->GetScale()
			);
	}
	if (m_input->IsKeyDown(GLFW_KEY_F))
	{
		m_graphics->GetCurrentlySelected()->Transform(
			m_graphics->GetCurrentlySelected()->GetPosition(),
			&((*m_graphics->GetCurrentlySelected()->GetRotation()) + vec3(-ROTATION_AMOUNT*m_timer->GetDeltaTime(), 0.0f, 0.0f)),
			m_graphics->GetCurrentlySelected()->GetScale()
			);
	}
	if (m_input->IsKeyDown(GLFW_KEY_G))
	{
		m_graphics->GetCurrentlySelected()->Transform(
			m_graphics->GetCurrentlySelected()->GetPosition(),
			&((*m_graphics->GetCurrentlySelected()->GetRotation()) + vec3(0.0f, -ROTATION_AMOUNT*m_timer->GetDeltaTime(), 0.0f)),
			m_graphics->GetCurrentlySelected()->GetScale()
			);
	}
	if (m_input->IsKeyDown(GLFW_KEY_H))
	{
		m_graphics->GetCurrentlySelected()->Transform(
			m_graphics->GetCurrentlySelected()->GetPosition(),
			&((*m_graphics->GetCurrentlySelected()->GetRotation()) + vec3(0.0f, 0.0f, -ROTATION_AMOUNT*m_timer->GetDeltaTime())),
			m_graphics->GetCurrentlySelected()->GetScale()
			);
	}

	// process mouse to rotate camera
	if (m_input->IsMouseButtonDown(GLFW_MOUSE_BUTTON_2))
	{
		glfwSetInputMode(m_graphics->GetWindowPtr(), GLFW_CURSOR, GLFW_CURSOR_DISABLED);

		ProcessCameraMovement();

		// reset mouse for the next frame
		glfwSetCursorPos(m_graphics->GetWindowPtr(), WINDOW_WIDTH / 2, WINDOW_HEIGHT / 2);
	}
	else
	{
		glfwSetInputMode(m_graphics->GetWindowPtr(), GLFW_CURSOR, GLFW_CURSOR_NORMAL);
	}

	if (m_input->IsMouseButtonDown(GLFW_MOUSE_BUTTON_1))
	{
		if (m_timer->GetCurrentTime() - m_timer->GetFlag(0) > 1000)
		{
			m_timer->SetFlag(0);
			ProcessMouseClick();
		}
	}
}

void System::ProcessCameraMovement()
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

void System::ProcessMouseClick()
{
	glfwGetCursorPos(m_graphics->GetWindowPtr(), &m_input->mouseX, &m_input->mouseY);

	m_graphics->RayCastAndSelect(m_input->mouseX, m_input->mouseY);
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
