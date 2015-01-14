#include "LightAmbient.h"

LightAmbient::LightAmbient() : Light()
{
	m_diffuseColor.x = 0.0f;
	m_diffuseColor.y = 0.0f;
	m_diffuseColor.z = 0.0f;
	m_diffuseColor.w = 1.0f;

	this->name = "LightAmbient";
}

LightAmbient::LightAmbient(D3DXVECTOR4 diffuseColor) : LightAmbient()
{
	m_diffuseColor = diffuseColor;
}

LightAmbient::LightAmbient(ifstream &is) : LightAmbient()
{
	string line; 

	is >> this->name;
	is >> this->m_diffuseColor.x;
	is >> this->m_diffuseColor.y;
	is >> this->m_diffuseColor.z;

	is >> line;
}

LightAmbient::LightAmbient(const LightAmbient&)
{

}

LightAmbient::~LightAmbient()
{
}

void LightAmbient::SetDiffuseColor(D3DXVECTOR4 diffuseColor)
{
	m_diffuseColor = diffuseColor;
}

void LightAmbient::SetDirection(D3DXVECTOR3 direction)
{
	// do nothing
}

void LightAmbient::SetPosition(D3DXVECTOR3 position)
{
	// do nothing
}

D3DXVECTOR4 LightAmbient::GetDiffuseColor()
{
	return m_diffuseColor;
}

D3DXVECTOR3 LightAmbient::GetDirection()
{
	return D3DXVECTOR3(0.0f, 0.0f, 0.0f);
}

D3DXVECTOR3 LightAmbient::GetPosition()
{
	return D3DXVECTOR3(0.0f, 0.0f, 0.0f);
}
