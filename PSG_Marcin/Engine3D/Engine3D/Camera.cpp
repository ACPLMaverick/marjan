#include "Camera.h"


Camera::Camera()
{
	m_position = D3DXVECTOR3(0.0f, 0.0f, 1.8f);
	m_rotation = D3DXVECTOR3(0.0f, 0.0f, 0.0f);
	m_target = D3DXVECTOR3(0.0f, 0.0f, 1.0f);
}

Camera::Camera(const Camera& other)
{

}

Camera::~Camera()
{
}

void Camera::SetPosition(D3DXVECTOR3 vec)
{
	m_position = vec;
}

void Camera::SetRotation(D3DXVECTOR3 vec)
{
	m_rotation = vec;
}

void Camera::SetTarget(D3DXVECTOR3 vec)
{
	m_target = vec;
}

D3DXVECTOR3 Camera::GetPosition()
{
	return m_position;
}

D3DXVECTOR3 Camera::GetRotation()
{
	return m_rotation;
}

D3DXVECTOR3 Camera::GetTarget()
{
	return m_target;
}

void Camera::Render()
{
	D3DXVECTOR3 up, position, lookAt;
	float yaw, pitch, roll;
	float rad = 0.0174532925f;
	D3DXMATRIX rotationMatrix;

	// vector that points upwards
	up.x = 0.0f;
	up.y = 1.0f;
	up.z = 0.0f;

	// position of camera in world
	position = m_position;

	// look at default position
	lookAt = m_target;

	// set the yaw (Y axis), pitch (X axis), and roll (Z axis) rotations in radians
	pitch = m_rotation.x * rad;
	yaw = m_rotation.y * rad;
	roll = m_rotation.z * rad;

	//create rotation matrix
	D3DXMatrixRotationYawPitchRoll(&rotationMatrix, yaw, pitch, roll);

	//transform lookAt and up Vector by rotation matrix so it's correctly rotated at origin
	D3DXVec3TransformCoord(&lookAt, &lookAt, &rotationMatrix);
	D3DXVec3TransformCoord(&up, &up, &rotationMatrix);

	//translate rotated camera position to the position of the viewer
	lookAt = lookAt + position;

	// CREATE VIEW MATRIX!
	D3DXMatrixLookAtLH(&m_viewMatrix, &position, &lookAt, &up);
}


void Camera::GetViewMatrix(D3DXMATRIX& viewMatrix)
{
	viewMatrix = m_viewMatrix;
}