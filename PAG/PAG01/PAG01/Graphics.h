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
#include "MeshManager.h"

#define GLFW_SAMPLES_VALUE 4
#define WINDOW_WIDTH 1280
#define WINDOW_HEIGHT 720
#define WINDOW_NAME "PAG01"
#define WINDOW_FOV 70.0f
#define WINDOW_NEAR 0.1f
#define WINDOW_FAR 1000.0f
static const float WINDOW_RATIO = ((float)WINDOW_WIDTH / (float)WINDOW_HEIGHT);

class Graphics
{
private:
	GLFWwindow* m_window;
	MeshManager* m_manager;
	Mesh* m_mesh;
	Mesh* test;
	Camera* m_camera;
	Light* m_light;

	GLuint programID;

	GLuint m_vertexBuffer;

	mat4 projectionMatrix;

	GLuint LoadShaders(const char* vertexFilePath, const char* fragmentFilePath);
	Mesh* SearchMeshTree(Mesh* node, vec3* ray);
public:
	Graphics();
	~Graphics();

	bool Initialize();
	void Shutdown();
	void Frame();

	GLFWwindow* GetWindowPtr();
	Camera* GetCameraPtr();

	Mesh* GetCurrentlySelected();
	void RayCastAndSelect(double mX, double mY);
};

