#include "LightPoint.h"


LightPoint::LightPoint() : Light()
{
	m_diffuseColor.x = 0.0f;
	m_diffuseColor.y = 0.0f;
	m_diffuseColor.z = 0.0f;
	m_diffuseColor.w = 1.0f;
	m_position.x = 0.0f;
	m_position.y = 0.0f;
	m_position.z = 0.0f;
	m_attenuation = 0.0f;

	this->name = "LightPoint";
}

LightPoint::LightPoint(D3DXVECTOR4 diffuseColor, D3DXVECTOR3 position, float attenuation) : LightPoint()
{
	m_diffuseColor = diffuseColor;
	m_position = position;
	m_attenuation = attenuation;
}

LightPoint::LightPoint(ifstream &is) : LightPoint()
{
	string line;

	is >> this->name;
	is >> this->m_diffuseColor.x;
	is >> this->m_diffuseColor.y;
	is >> this->m_diffuseColor.z;
	is >> this->m_position.x;
	is >> this->m_position.y;
	is >> this->m_position.z;
	is >> this->m_attenuation;

	is >> line;
}

LightPoint::LightPoint(const LightPoint&)
{

}

LightPoint::~LightPoint()
{
}

void LightPoint::SetDiffuseColor(D3DXVECTOR4 diffuseColor)
{
	m_diffuseColor = diffuseColor;
}

void LightPoint::SetDirection(D3DXVECTOR3 direction)
{
	m_position = direction;
}

void LightPoint::SetPosition(D3DXVECTOR3 position)
{
	// do nothing
}

D3DXVECTOR4 LightPoint::GetDiffuseColor()
{
	return m_diffuseColor;
}

D3DXVECTOR3 LightPoint::GetDirection()
{
	return D3DXVECTOR3(0.0f, 0.0f, 0.0f);
}

D3DXVECTOR3 LightPoint::GetPosition()
{
	return m_position;
}

void LightPoint::SetAttenuation(float att)
{
	m_attenuation = att;
}

float LightPoint::GetAttenuation()
{
	return m_attenuation;
}