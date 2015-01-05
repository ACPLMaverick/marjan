#pragma once
#include <D3DX10math.h>
#include <iostream>

class Light
{
private:

public:
	std::string name;

	Light();
	Light(const Light&);
	~Light();

	virtual void SetDiffuseColor(D3DXVECTOR4 diffuseColor) = 0;
	virtual void SetDirection(D3DXVECTOR3 direction) = 0;
	virtual void SetPosition(D3DXVECTOR3 position) = 0;

	virtual D3DXVECTOR4 GetDiffuseColor() = 0;
	virtual D3DXVECTOR3 GetDirection() = 0;
	virtual D3DXVECTOR3 GetPosition() = 0;
};

