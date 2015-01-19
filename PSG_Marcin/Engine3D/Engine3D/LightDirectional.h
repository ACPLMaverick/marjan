#pragma once
#include "Light.h"

class LightDirectional : public Light
{
private:
	D3DXVECTOR4 m_diffuseColor;
	D3DXVECTOR3 m_direction;
public:
	LightDirectional();
	LightDirectional(D3DXVECTOR4 diffuseColor, D3DXVECTOR3 direction);
	LightDirectional(ifstream &is);
	LightDirectional(const LightDirectional&);
	~LightDirectional();

	virtual void SetDiffuseColor(D3DXVECTOR4 diffuseColor);
	virtual void SetDirection(D3DXVECTOR3 direction);
	virtual void SetPosition(D3DXVECTOR3 position);

	virtual D3DXVECTOR4 GetDiffuseColor();
	virtual D3DXVECTOR3 GetDirection();
	virtual D3DXVECTOR3 GetPosition();
};

