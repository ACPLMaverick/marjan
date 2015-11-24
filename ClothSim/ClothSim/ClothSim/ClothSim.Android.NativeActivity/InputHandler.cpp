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

void InputHandler::GetCursorPosition(glm::vec2* vec)
{
	InputManager::GetInstance()->GetTouchPosition(vec);
}

void InputHandler::GetCursorVector(glm::vec2* vec)
{
	InputManager::GetInstance()->GetTouchDirection(vec);
}

void InputHandler::GetArrowsMovementVector(glm::vec3* vec)
{
	/*glm::vec3 vec = glm::vec3(0.0f, 0.0f, 0.0f);

	if (InputManager::GetInstance()->GetKey(GLFW_KEY_W))
		vec.z = -1.0f;
	if (InputManager::GetInstance()->GetKey(GLFW_KEY_S))
		vec.z = 1.0f;
	if (InputManager::GetInstance()->GetKey(GLFW_KEY_A))
		vec.x = -1.0f;
	if (InputManager::GetInstance()->GetKey(GLFW_KEY_D))
		vec.x = 1.0f;
	if (InputManager::GetInstance()->GetKey(GLFW_KEY_Q))
		vec.y = 1.0f;
	if (InputManager::GetInstance()->GetKey(GLFW_KEY_Z))
		vec.y = -1.0f;

	if (vec.x != 0.0f || vec.y != 0.0f || vec.z != 0.0f)
	vec = glm::normalize(vec);

	return vec;
	*/
	glm::vec2 r;
	InputManager::GetInstance()->GetTouchPosition(&r);
	vec->x = r.x;
	vec->y = 0.0f;
	vec->z = -r.y;
}

int InputHandler::GetZoomValue()
{
	glm::vec4 swipe;
	glm::vec2 swipe1, swipe2;
	InputManager::GetInstance()->GetSwipeDirections(&swipe);

	swipe1.x = swipe.x;
	swipe1.y = swipe.y;
	swipe2.x = swipe.z;
	swipe2.y = swipe.w;

	float dot = glm::dot(swipe1, swipe2);

	return (int)dot;
}

bool InputHandler::ActionButtonPressed()
{
	return InputManager::GetInstance()->GetDoubleTouch();
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
