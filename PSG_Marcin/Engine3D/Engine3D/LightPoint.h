#pragma once
#include "Light.h"

class LightPoint : public Light
{
private:
	D3DXVECTOR4 m_diffuseColor;
	D3DXVECTOR3 m_position;
	float m_attenuation;
public:
	LightPoint();
	LightPoint(D3DXVECTOR4 diffuseColor, D3DXVECTOR3 position, float attenuation);
	LightPoint(ifstream &is);
	LightPoint(const LightPoint&);
	~LightPoint();

	virtual void SetDiffuseColor(D3DXVECTOR4 diffuseColor);
	virtual void SetDirection(D3DXVECTOR3 direction);
	virtual void SetPosition(D3DXVECTOR3 position);
	virtual void SetAttenuation(float att);

	virtual D3DXVECTOR4 GetDiffuseColor();
	virtual D3DXVECTOR3 GetDirection();
	virtual D3DXVECTOR3 GetPosition();
	virtual float GetAttenuation();
};

