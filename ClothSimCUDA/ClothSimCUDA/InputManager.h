#pragma once

/*
	This class controls all input devices and provides an interface for InputHandler to read input device status.
*/

#include "Common.h"
#include "Renderer.h"

#include <GL\glew.h>
#include <GLFW\glfw3.h>

struct MouseData
{
	double mouseX, mouseY, mouseXRelative, mouseYRelative;
	int scroll, scrollRelative;

	MouseData()
	{
		mouseX = mouseY = mouseXRelative = mouseYRelative = 0.0;
		scroll = scrollRelative = 0;
	}
};

class InputManager
{
protected:
	static InputManager* instance;
	InputManager();

	MouseData m_mouseData;
	int m_scrollHelper = 0;

	static void MouseScrollCallback(GLFWwindow*, double, double);
public:
	InputManager(const InputManager*);
	~InputManager();

	static InputManager* GetInstance();
	static void DestroyInstance();

	unsigned int Initialize();
	unsigned int Shutdown();
	unsigned int Run();

	bool GetKey(int);
	bool GetKeyDown(int);
	bool GetKeyUp(int);

	MouseData* GetMouseData();
};

