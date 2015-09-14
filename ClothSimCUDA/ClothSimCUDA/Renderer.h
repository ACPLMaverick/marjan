#pragma once

#include "Common.h"
#include "Singleton.h"
#include "Settings.h"
#include "System.h"
#include "ResourceManager.h"

#include <GL\glew.h>
#include <GLFW\glfw3.h>
#include <glm\glm\glm.hpp>
#include <SOIL2\SOIL2.h>
#include <fstream>
#include <string>
#include <vector>

using namespace std;

class System;
class ResourceManager;

enum DrawMode { BASIC, WIREFRAME, BASIC_WIREFRAME };

class Renderer : public Singleton<Renderer>
{
	friend class Singleton<Renderer>;

protected:
	Renderer();

	DrawMode m_mode;

	GLFWwindow* m_window;
	ShaderID* m_shaderID;

	string sn_nameBasic = "Basic";
	string sn_nameWf = "Wireframe";
public:
	Renderer(const Renderer*);
	~Renderer();

	unsigned int Initialize();
	unsigned int Shutdown();
	unsigned int Run();

	void SetDrawMode(DrawMode mode);

	ShaderID* GetCurrentShaderID();
	GLFWwindow* GetWindow();
	DrawMode GetDrawMode();

	static void LoadShaders(const string*, const string*, const string*, ShaderID*);
	static void ShutdownShader(ShaderID*);
	static void LoadTexture(const string*, TextureID*);
	static void LoadTexture(const string*, const unsigned char*, int, int, int, int, TextureID*);
	static void ShutdownTexture(TextureID*);
};

