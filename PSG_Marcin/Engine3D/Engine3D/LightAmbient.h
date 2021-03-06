#pragma once
#include "Light.h"

class LightAmbient : public Light
{
private:
	D3DXVECTOR4 m_diffuseColor;
	D3DXVECTOR3 m_direction;
public:
	LightAmbient();
	LightAmbient(D3DXVECTOR4 diffuseColor);
	LightAmbient(ifstream &is);
	LightAmbient(const LightAmbient&);
	~LightAmbient();

	virtual void SetDiffuseColor(D3DXVECTOR4 diffuseColor);
	virtual void SetDirection(D3DXVECTOR3 direction);
	virtual void SetPosition(D3DXVECTOR3 position);

	virtual D3DXVECTOR4 GetDiffuseColor();
	virtual D3DXVECTOR3 GetDirection();
	virtual D3DXVECTOR3 GetPosition();
};

