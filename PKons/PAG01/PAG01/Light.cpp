#include "Light.h"


Light::Light(glm::vec4* lightDirection, glm::vec4* lightDiffuse, glm::vec4* lightSpecular, glm::vec4* lightAmbient, GLfloat glossiness, GLuint programID)
{
	m_programID = programID;

	this->lightDiffuse = *lightDiffuse;
	this->lightAmbient = *lightAmbient;
	this->lightSpecular = *lightSpecular;
	this->glossiness = glossiness;

	this->lightDirection = glm::normalize(*lightDirection);
	
	lightDirID = glGetUniformLocation(m_programID, "lightDirection");
	lightDifID = glGetUniformLocation(m_programID, "lightDiffuse");
	lightSpecID = glGetUniformLocation(m_programID, "lightSpecular");
	lightAmbID = glGetUniformLocation(m_programID, "lightAmbient");
	glossID = glGetUniformLocation(m_programID, "glossiness");
}


Light::~Light()
{
}
