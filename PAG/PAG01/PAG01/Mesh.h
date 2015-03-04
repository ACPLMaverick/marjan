#pragma once
#include <GL\glew.h>
#include <GLFW\glfw3.h>
#include <glm\glm\glm.hpp>

class Mesh
{
private:
	GLuint m_vertexArrayID;
	GLfloat* g_vertex_buffer_data;
	unsigned int g_vertex_count;
	GLuint m_vertexBuffer;
public:
	Mesh();
	~Mesh();

	bool Initialize();
	void Shutdown();

	void Draw();
};

