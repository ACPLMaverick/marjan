#include "Input.h"


Input::Input()
{
}


Input::~Input()
{
}

bool Input::Initialize(GLFWwindow* window)
{
	m_window = window;
	glfwSetInputMode(m_window, GLFW_STICKY_KEYS, GL_TRUE);
	glfwSetInputMode(m_window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

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