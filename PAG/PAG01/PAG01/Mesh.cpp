#include "Mesh.h"


Mesh::Mesh()
{
}


Mesh::~Mesh()
{
}

bool Mesh::Initialize()
{
	glGenVertexArrays(1, &m_vertexArrayID);
	glBindVertexArray(m_vertexArrayID);
 
	g_vertex_count = 3;
	g_vertex_buffer_data = new GLfloat[3*g_vertex_count] {
		-1.0f, -1.0f, 0.0f,
		1.0f, -1.0f, 0.0f,
		0.0f, 1.0f, 0.0f
	};

	glGenBuffers(1, &m_vertexBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, m_vertexBuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(g_vertex_buffer_data[0])*3*g_vertex_count, g_vertex_buffer_data,
		GL_STATIC_DRAW);

	return true;
}

void Mesh::Shutdown()
{
	delete[] g_vertex_buffer_data;

	glDeleteBuffers(1, &m_vertexBuffer);
	glDeleteVertexArrays(1, &m_vertexArrayID);
}

void Mesh::Draw()
{
	glBindBuffer(GL_ARRAY_BUFFER, m_vertexBuffer);
	glVertexAttribPointer(
		0,
		g_vertex_count,
		GL_FLOAT,
		GL_FALSE,
		0,
		(void*)0
		);
	glDrawArrays(GL_TRIANGLES, 0, g_vertex_count);
	return;
}
