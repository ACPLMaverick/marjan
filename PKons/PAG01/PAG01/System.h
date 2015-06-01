#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <GL\glew.h>
#include <GLFW\glfw3.h>
#include <glm\glm\glm.hpp>
#include <Windows.h>
using namespace glm;
using namespace std;

#include "Graphics.h"
#include "Input.h"
#include "Timer.h"

#define TRANSLATION_AMOUNT 0.01f
#define ROTATION_AMOUNT 0.01f

class System
{
private:
	static System* me;
	Input* m_input;
	Timer* m_timer;

	bool isRunning;
	void ProcessInput();
	void ProcessCameraMovement();
	void ProcessMouseClick();
	static void MouseScrollCallback(GLFWwindow* window, double x, double y);
	System();
public:
	static System* GetInstance();
	static void DestroyInstance();
	~System();

	void Initialize();
	void Shutdown();
	void GameLoop();
};

