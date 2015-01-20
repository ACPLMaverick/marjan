#include "Camera.h"


Camera::Camera()
{
	m_position = D3DXVECTOR3(0.0f, 1.0f, 1.8f);
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

	m_yawPitchRoll = D3DXVECTOR3(yaw, pitch, roll);

	//create rotation matrix
	D3DXMatrixRotationYawPitchRoll(&rotationMatrix, yaw, pitch, roll);

	//transform lookAt and up Vector by rotation matrix so it's correctly rotated at origin
	D3DXVec3TransformCoord(&lookAt, &lookAt, &rotationMatrix);
	D3DXVec3TransformCoord(&up, &up, &rotationMatrix);

	//translate rotated camera position to the position of the viewer
	lookAt = lookAt + position;

	// CREATE VIEW MATRIX!
	D3DXMatrixLookAtLH(&m_viewMatrix, &m_position, &lookAt, &up);
}

void Camera::RenderBaseViewMatrix()
{
	D3DXVECTOR3 up, position, lookAt;
	float yaw, pitch, roll;
	D3DXMATRIX rotationMatrix;


	// Setup the vector that points upwards.
	up.x = 0.0f;
	up.y = 1.0f;
	up.z = 0.0f;

	// Setup the position of the camera in the world.
	position.x = m_position.x;
	position.y = m_position.y;
	position.z = m_position.z;

	// Setup where the camera is looking by default.
	lookAt.x = 0.0f;
	lookAt.y = 0.0f;
	lookAt.z = 1.0f;

	// Set the yaw (Y axis), pitch (X axis), and roll (Z axis) rotations in radians.
	pitch = m_rotation.x * 0.0174532925f;
	yaw = m_rotation.y * 0.0174532925f;
	roll = m_rotation.z * 0.0174532925f;

	// Create the rotation matrix from the yaw, pitch, and roll values.
	D3DXMatrixRotationYawPitchRoll(&rotationMatrix, yaw, pitch, roll);

	// Transform the lookAt and up vector by the rotation matrix so the view is correctly rotated at the origin.
	D3DXVec3TransformCoord(&lookAt, &lookAt, &rotationMatrix);
	D3DXVec3TransformCoord(&up, &up, &rotationMatrix);

	// Translate the rotated camera position to the location of the viewer.
	lookAt = position + lookAt;

	// Finally create the base view matrix from the three updated vectors.
	D3DXMatrixLookAtLH(&m_baseViewMatrix, &position, &lookAt, &up);
}


void Camera::GetViewMatrix(D3DXMATRIX& viewMatrix)
{
	viewMatrix = m_viewMatrix;
}

void Camera::GetBaseViewMatrix(D3DXMATRIX& baseViewMatrix)
{
	baseViewMatrix = m_baseViewMatrix;
}

D3DXVECTOR3 Camera::GetYawPitchRoll()
{
	return m_yawPitchRoll;
}