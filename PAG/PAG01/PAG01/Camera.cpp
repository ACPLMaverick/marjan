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
	target = glm::vec3(0.0f, 0.0f, 0.0f);
	up = glm::vec3(0.0f, 1.0f, 0.0f);
	right = glm::vec3(1.0f, 0.0f, 0.0f);
	viewMatrix = glm::lookAt(position, target, up);

	return true;
}

void Camera::Shutdown()
{

}

void Camera::Transform(glm::vec3* position, glm::vec3* target, glm::vec3* up, glm::vec3* right)
{
	this->position = *position;
	this->target = *target;
	this->up = *up;
	this->right = *right;
	
	glm::mat4 translation, target_x, target_y, target_z;
	translation = glm::translate(*position);
	target_x = glm::rotate((*target).x, glm::vec3(1.0f, 0.0f, 0.0f));
	target_y = glm::rotate((*target).y, glm::vec3(0.0f, 1.0f, 0.0f));
	target_z = glm::rotate((*target).z, glm::vec3(0.0f, 0.0f, 1.0f));

	viewMatrix = glm::lookAt(*position, *target, this->up);
}

glm::mat4* Camera::GetViewMatrix()
{
	return &(this->viewMatrix);
}

glm::vec3 Camera::GetPosition()
{
	return this->position;
}

glm::vec3 Camera::GetTarget()
{
	return this->target;
}

glm::vec3 Camera::GetEyeVector()
{
	glm::vec3 vec = (this->target - this->position);
	//glm::vec4 mul = glm::vec4(vec.x, vec.y, vec.z, 1.0f);
	//mul = mul * viewMatrix;
	//vec.x = mul.x; vec.y = mul.y; vec.z = mul.z;
	return glm::normalize(vec);
}

glm::vec3 Camera::GetUp()
{
	return this->up;
}

glm::vec3 Camera::GetRight()
{
	return this->right;
}