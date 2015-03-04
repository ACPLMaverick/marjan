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
	Graphics* m_graphics;
	Input* m_input;

	bool isRunning;
public:
	System();
	~System();

	void Initialize();
	void Shutdown();
	void GameLoop();
};

