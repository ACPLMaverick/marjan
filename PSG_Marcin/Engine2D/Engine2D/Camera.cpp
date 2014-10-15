#include "Camera.h"


Camera::Camera()
{
	m_positionX = 0.0f;
	m_positionY = 0.0f;
	m_positionZ = 0.0f;

	m_rotationX = 0.0f;
	m_rotationY = 0.0f;
	m_rotationZ = 0.0f;
}

Camera::Camera(const Camera& other)
{

}

Camera::~Camera()
{
}

void Camera::SetPosition(float x, float y, float z)
{
	m_positionX = x;
	m_positionY = y;
	m_positionZ = z;
}

void Camera::SetRotation(float x, float y, float z)
{
	m_rotationX = x;
	m_rotationY = y;
	m_rotationZ = z;
}

D3DXVECTOR3 Camera::GetPosition()
{
	return D3DXVECTOR3(m_positionX, m_positionY, m_positionZ);
}

D3DXVECTOR3 Camera::GetRotation()
{
	return D3DXVECTOR3(m_positionX, m_positionY, m_positionZ);
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
	position.x = m_positionX;
	position.y = m_positionY;
	position.z = m_positionZ;

	// look at default position
	lookAt.x = 0.0f;
	lookAt.y = 0.0f;
	lookAt.z = 1.0f;

	// set the yaw (Y axis), pitch (X axis), and roll (Z axis) rotations in radians
	pitch = m_rotationX * rad;
	yaw = m_rotationY * rad;
	roll = m_rotationZ * rad;

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