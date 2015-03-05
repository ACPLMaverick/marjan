#include "Camera.h"


Camera::Camera()
{
}


Camera::~Camera()
{
}

bool Camera::Initialize()
{
	position = glm::vec3(0.0f, 0.0f, 0.0f);
	rotation = glm::vec3(0.0f, 0.0f, 0.0f);
	up = glm::vec3(0.0f, 1.0f, 0.0f);
	viewMatrix = glm::lookAt(position, rotation, up);

	return true;
}

void Camera::Shutdown()
{

}

void Camera::Transform(glm::vec3 position, glm::vec3 rotation)
{
	this->position = position;
	this->rotation = rotation;
	
	glm::mat4 translation, rotation_x, rotation_y, rotation_z;
	translation = glm::translate(position);
	rotation_x = glm::rotate(rotation.x, glm::vec3(1.0f, 0.0f, 0.0f));
	rotation_y = glm::rotate(rotation.y, glm::vec3(0.0f, 1.0f, 0.0f));
	rotation_z = glm::rotate(rotation.z, glm::vec3(0.0f, 0.0f, 1.0f));

	viewMatrix = glm::lookAt(position, rotation, this->up);
}

glm::mat4* Camera::GetViewMatrix()
{
	return &(this->viewMatrix);
}

glm::vec3 Camera::GetPosition()
{
	return this->position;
}

glm::vec3 Camera::GetRotation()
{
	return this->rotation;
}

glm::vec3 Camera::GetUp()
{
	return this->up;
}