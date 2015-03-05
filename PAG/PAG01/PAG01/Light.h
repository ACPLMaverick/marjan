#pragma once
#include <GL\glew.h>
#include <GLFW\glfw3.h>
#include <glm\glm\glm.hpp>
#include <glm\glm\gtx\transform.hpp>

class Light
{
private:
	GLuint m_programID;
public:
	glm::vec4 lightDirection;
	glm::vec4 lightDiffuse;
	glm::vec4 lightSpecular;
	glm::vec4 lightAmbient;
	GLfloat glossiness;
	GLuint lightDirID, lightDifID, lightSpecID, lightAmbID, glossID;

	Light(glm::vec4*, glm::vec4*, glm::vec4*, glm::vec4*, GLfloat, GLuint);
	~Light();
};

