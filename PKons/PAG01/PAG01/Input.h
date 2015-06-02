#pragma once

#include <GLFW\glfw3.h>

#define MOUSE_SPEED 0.01f
#define SCROLL_SPEED 0.1f

class Input
{
private:
	GLFWwindow* m_window;
public:
	double mouseX, mouseY;
	float horizontalAngle, verticalAngle, horizontalActual, verticalActual;

	Input();
	~Input();

	bool Initialize(GLFWwindow* window, GLFWscrollfun fun);
	void Shutdown();

	bool IsKeyDown(int keyCode);
	bool IsMouseButtonDown(int keyCode);
};

