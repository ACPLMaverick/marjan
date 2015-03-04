#pragma once

#include <GLFW\glfw3.h>

class Input
{
private:
	GLFWwindow* m_window;
public:
	Input();
	~Input();

	bool Initialize(GLFWwindow* window);
	void Shutdown();

	bool IsKeyDown(int keyCode);
};

