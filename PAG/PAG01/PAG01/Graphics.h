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
#include "Camera.h"
#include "Light.h"

#define GLFW_SAMPLES_VALUE 4
#define WINDOW_WIDTH 1024
#define WINDOW_HEIGHT 768
#define WINDOW_NAME "PAG01"
#define WINDOW_FOV 45.0f
#define WINDOW_NEAR 0.1f
#define WINDOW_FAR 100.0f
static const float WINDOW_RATIO = ((float)WINDOW_WIDTH / (float)WINDOW_HEIGHT);

static const string PATH_DIFFUSE = "E:\\_projects\\Engine2DAssets\\water01.dds";
static const string PATH_SPECULAR = "E:\\_projects\\Engine2DAssets\\water02.dds";

class Graphics
{
private:
	GLFWwindow* m_window;
	Mesh* m_mesh;
	Texture* m_texture;
	Camera* m_camera;
	Light* m_light;

	GLuint programID;

	GLuint m_vertexBuffer;

	mat4 projectionMatrix;

	GLuint LoadShaders(const char* vertexFilePath, const char* fragmentFilePath);
public:
	Graphics();
	~Graphics();

	bool Initialize();
	void Shutdown();
	void Frame();

	GLFWwindow* GetWindowPtr();
	Camera* GetCameraPtr();
};

