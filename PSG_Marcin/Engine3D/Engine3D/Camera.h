#pragma once

//includes
#include <D3DX10math.h>

class Camera
{
private:
	D3DXVECTOR3 m_position;
	D3DXVECTOR3 m_rotation;
	D3DXVECTOR3 m_target;

	D3DXMATRIX m_viewMatrix;
public:
	Camera();
	Camera(const Camera&);
	~Camera();

	void SetPosition(D3DXVECTOR3 vec);
	void SetRotation(D3DXVECTOR3 vec);
	void SetTarget(D3DXVECTOR3 vec);

	D3DXVECTOR3 GetPosition();
	D3DXVECTOR3 GetRotation();
	D3DXVECTOR3 GetTarget();

	void Render();
	void GetViewMatrix(D3DXMATRIX&);
};

