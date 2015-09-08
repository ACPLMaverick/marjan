#pragma once

#include "Common.h"
#include "Settings.h"
#include "System.h"

#include <GL\glew.h>
#include <GLFW\glfw3.h>
#include <glm\glm\glm.hpp>
#include <fstream>
#include <string>
#include <vector>

using namespace std;

class System;

struct ShaderID
{
	int id;
	string name;

	int id_worldViewProj;
	int id_world;
	int id_worldInvTrans;
	int id_eyeVector;
	int id_lightDir;
	int id_lightDiff;
	int id_lightSpec;
	int id_lightAmb;
	int id_gloss;
	int id_highlight;
};

enum DrawMode { BASIC, WIREFRAME, BASIC_WIREFRAME };

class Renderer
{
protected:
	static Renderer* instance;
	Renderer();

	DrawMode m_mode;

	GLFWwindow* m_window;
	ShaderID m_shaderID;
	vector<ShaderID> m_shaders;

	ShaderID LoadShaders(const char*, const char*, const string*);
public:
	Renderer(const Renderer*);
	~Renderer();

	static Renderer* GetInstance();
	static void DestroyInstance();

	unsigned int Initialize();
	unsigned int Shutdown();
	unsigned int Run();

	void SetDrawMode(DrawMode mode);

	ShaderID* GetCurrentShaderID();
	ShaderID* GetShaderIDByName(const string*);
	GLFWwindow* GetWindow();
	DrawMode GetDrawMode();
};

