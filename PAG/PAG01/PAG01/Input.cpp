#include "Input.h"


Input::Input()
{
	horizontalAngle = 0.0f;
	verticalAngle = 0.0f;
	horizontalActual = 0.0f;
	verticalActual = 0.0f;
}


Input::~Input()
{
}

bool Input::Initialize(GLFWwindow* window)
{
	m_window = window;
	glfwSetInputMode(m_window, GLFW_STICKY_KEYS, GL_TRUE);
	glfwSetInputMode(m_window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
	glfwSetInputMode(m_window, GLFW_STICKY_MOUSE_BUTTONS, GL_TRUE);

	return true;
}

void Input::Shutdown()
{

}

bool Input::IsKeyDown(int keyCode)
{
	if (glfwGetKey(m_window, keyCode) == GLFW_PRESS) return true;
	else return false;
}

bool Input::IsMouseButtonDown(int keyCode)
{
	if (glfwGetMouseButton(m_window, keyCode) == GLFW_PRESS) return true;
	else return false;
}