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
	// EXITING

	if (InputHandler::GetInstance()->ExitPressed())
	{
		System::GetInstance()->Stop();
	}

	///////////////////////////

	// ROTATING CAMERA

	if (InputHandler::GetInstance()->CameraRotateButtonPressed())
	{
#ifdef BUILD_OPENGL
		if (!cursorHideHelper)
		{
			glfwSetInputMode(Renderer::GetInstance()->GetWindow(), GLFW_CURSOR, GLFW_CURSOR_HIDDEN);
			cursorHideHelper = true;
		}
#endif
		glm::vec2 camVec = InputHandler::GetInstance()->GetCursorVector();
		glm::vec4 camCurrentPos = glm::vec4(*System::GetInstance()->GetCurrentScene()->GetCamera()->GetPosition(), 1.0f);
		glm::vec3 camRight = *System::GetInstance()->GetCurrentScene()->GetCamera()->GetRight();

		glm::mat4 horRotation = glm::rotate(camVec.x * CSSET_CAMERA_ROTATE_SPEED, glm::vec3(0.0f, 1.0f, 0.0f));
		glm::mat4 vertRotation = glm::rotate(camVec.y * CSSET_CAMERA_ROTATE_SPEED, camRight);

		camCurrentPos = camCurrentPos * horRotation * vertRotation;
		glm::vec3 newPos = glm::vec3(camCurrentPos.x, camCurrentPos.y, camCurrentPos.z);
		
		glm::vec3 dir = *System::GetInstance()->GetCurrentScene()->GetCamera()->GetTarget() - newPos;
		glm::vec3 dirYZero = glm::vec3(dir.x, 0.0f, dir.z);
		float dot = glm::dot(dir, dirYZero);

		if (dot > CSSET_CAMERA_ROTATE_BARRIER)
		{
			System::GetInstance()->GetCurrentScene()->GetCamera()->SetPosition(&newPos);
		}
	}
#ifdef BUILD_OPENGL
	else if (cursorHideHelper)
	{
		glfwSetInputMode(Renderer::GetInstance()->GetWindow(), GLFW_CURSOR, GLFW_CURSOR_NORMAL);
		glfwSetCursorPos(Renderer::GetInstance()->GetWindow(), CSSET_WINDOW_WIDTH / 2.0, CSSET_WINDOW_HEIGHT / 2.0);
		cursorHideHelper = false;
	}
#endif

	//////////////////////////

	// ZOOMING CAMERA

	int relScroll = InputHandler::GetInstance()->GetZoomValue();
	if (relScroll != 0)
	{
		glm::vec3 cPos = *System::GetInstance()->GetCurrentScene()->GetCamera()->GetPosition();
		float length = glm::length<float>(cPos);
		float scrollValue = (float)relScroll * CSSET_CAMERA_ZOOM_SPEED;
		scrollValue = length - scrollValue;
		
		if (scrollValue >= CSSET_CAMERA_ZOOM_BARRIER_MIN && scrollValue <= CSSET_CAMERA_ZOOM_BARRIER_MAX)
		{
			System::GetInstance()->GetCurrentScene()->GetCamera()->SetPosition(&(glm::normalize(cPos) * scrollValue));
		}
	}

	//////////////////////////

	// MOVING CAMERA ON XZ PLANE

	if (InputHandler::GetInstance()->CameraMoveButtonPressed())
	{
		glm::vec2 mVec = InputHandler::GetInstance()->GetCursorVector();
		if (mVec.x != 0.0f || mVec.y != 0.0f)
		{
			glm::vec3 cTarget = *System::GetInstance()->GetCurrentScene()->GetCamera()->GetTarget();
			glm::vec3 cPos = *System::GetInstance()->GetCurrentScene()->GetCamera()->GetPosition();
			glm::vec3 cDir = *System::GetInstance()->GetCurrentScene()->GetCamera()->GetDirection();
			glm::vec3 cRght = *System::GetInstance()->GetCurrentScene()->GetCamera()->GetRight();
			cDir.y = 0.0f;
			cRght.y = 0.0f;
			
			cTarget = cTarget + cRght * -mVec.x * CSSET_CAMERA_MOVE_SPEED;
			cTarget = cTarget + cDir * mVec.y * CSSET_CAMERA_MOVE_SPEED;

			if (glm::abs(cTarget.x) <= CSSET_CAMERA_POSITION_MAX && glm::abs(cTarget.z) <= CSSET_CAMERA_POSITION_MAX)
			{
				cPos = cPos + cRght * -mVec.x * CSSET_CAMERA_MOVE_SPEED;
				cPos = cPos + cDir * mVec.y * CSSET_CAMERA_MOVE_SPEED;
				System::GetInstance()->GetCurrentScene()->GetCamera()->SetPosition(&cPos);
				System::GetInstance()->GetCurrentScene()->GetCamera()->SetTarget(&cTarget);
			}
		}
	}

	//////////////////////////
	return CS_ERR_NONE;
}

unsigned int SimController::Draw()
{
	return CS_ERR_NONE;
}