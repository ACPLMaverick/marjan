#pragma once

#include <GL\glew.h>
#include <GLFW\glfw3.h>
#include <glm\glm\glm.hpp>
#include <glm\glm\gtx\transform.hpp>

class Camera
{
private:
	glm::mat4 viewMatrix;
	glm::vec3 position, target, up;
public:
	GLuint m_eyeVectorID;
	Camera();
	~Camera();

	bool Initialize();
	void Shutdown();

	void Transform(glm::vec3 position, glm::vec3 target);
	glm::mat4* GetViewMatrix();
	glm::vec3 GetPosition();
	glm::vec3 GetTarget();
	glm::vec3 GetUp();
	glm::vec3 GetEyeVector();
};

