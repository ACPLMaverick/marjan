#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <GL\glew.h>
#include <GLFW\glfw3.h>
#include <glm\glm\glm.hpp>
using namespace glm;
using namespace std;

#include "Graphics.h"
#include "Input.h"

class System
{
private:
	static System* me;
	Graphics* m_graphics;
	Input* m_input;

	bool isRunning;
	void ProcessInput();
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

