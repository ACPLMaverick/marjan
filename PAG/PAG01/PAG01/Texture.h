#pragma once
#include <GL\glew.h>
#include <GLFW\glfw3.h>
#include <glm\glm\glm.hpp>
#include <glm\glm\gtx\transform.hpp>
#include <iostream>
#include <SOIL2\SOIL2.h>
using namespace std;

class Texture
{
protected:
	GLuint m_ID;
	GLsizei m_texWidth, m_texHeight;

	virtual void LoadFromFile(const string* filePath);
public:
	Texture();
	~Texture();

	bool Initialize(const string* filePath);
	void Shutdown();

	GLuint GetID();
};

