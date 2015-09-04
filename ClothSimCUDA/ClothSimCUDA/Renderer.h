#pragma once

#include "Common.h"
#include "GraphicsSettings.h"
#include "System.h"

#include <GL\glew.h>
#include <GLFW\glfw3.h>
#include <glm\glm\glm.hpp>
#include <fstream>
#include <string>
#include <vector>

using namespace std;

class System;

class Renderer
{
protected:
	static Renderer* instance;
	Renderer();

	GLFWwindow* m_window;
	GLuint m_shaderID;


	GLuint LoadShaders(const char* vertexFilePath, const char* fragmentFilePath);
public:
	Renderer(const Renderer*);
	~Renderer();

	static Renderer* GetInstance();
	static void DestroyInstance();

	unsigned int Initialize();
	unsigned int Shutdown();
	unsigned int Run();

	GLuint GetCurrentShaderID();
};

