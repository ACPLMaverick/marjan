#include "LightDirectional.h"


LightDirectional::LightDirectional() : Light()
{
	m_diffuseColor.x = 0.0f;
	m_diffuseColor.y = 0.0f;
	m_diffuseColor.z = 0.0f;
	m_diffuseColor.w = 0.0f;
	m_direction.x = 0.0f;
	m_direction.y = 0.0f;
	m_direction.z = 0.0f;

	this->name = "LightDirectional";
}

LightDirectional::LightDirectional(D3DXVECTOR4 diffuseColor, D3DXVECTOR3 direction) : LightDirectional()
{
	m_diffuseColor = diffuseColor;
	m_direction = direction;
}

LightDirectional::LightDirectional(const LightDirectional&)
{

}

LightDirectional::~LightDirectional()
{
}

void LightDirectional::SetDiffuseColor(D3DXVECTOR4 diffuseColor)
{
	m_diffuseColor = diffuseColor;
}

void LightDirectional::SetDirection(D3DXVECTOR3 direction)
{
	m_direction = direction;
}

void LightDirectional::SetPosition(D3DXVECTOR3 position)
{
	// do nothing
}

D3DXVECTOR4 LightDirectional::GetDiffuseColor()
{
	return m_diffuseColor;
}

D3DXVECTOR3 LightDirectional::GetDirection()
{
	return m_direction;
}

D3DXVECTOR3 LightDirectional::GetPosition()
{
	return D3DXVECTOR3(0.0f, 0.0f, 0.0f);
}
