#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <GL\glew.h>
#include <GLFW\glfw3.h>
#include <glm\glm\glm.hpp>
#include <glm\glm\gtc\matrix_transform.hpp>
using namespace glm;
using namespace std;

#include "Mesh.h"

#define GLFW_SAMPLES_VALUE 4
#define WINDOW_WIDTH 1024
#define WINDOW_HEIGHT 768
#define WINDOW_NAME "PAG01"

class Graphics
{
private:
	GLFWwindow* m_window;
	Mesh* m_mesh;

	GLuint programID;
	GLuint matrixID;

	mat4 modelMatrix, viewMatrix, projectionMatrix, mvpMatrix;

	GLuint LoadShaders(const char* vertexFilePath, const char* fragmentFilePath);
public:
	Graphics();
	~Graphics();

	bool Initialize();
	void Shutdown();
	void Frame();

	GLFWwindow* GetWindowPtr();
};

