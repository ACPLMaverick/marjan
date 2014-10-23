#include "Bullet.h"


Bullet::Bullet() : GameObject()
{
}

Bullet::Bullet(const Bullet &other)
{
}

Bullet::Bullet(string name, string tag, Texture* texture, TextureShader* shader, ID3D11Device* device, D3DXVECTOR3 position, D3DXVECTOR3 rotation, D3DXVECTOR3 scale, float speed, float distance)
	: GameObject(name, tag, texture, shader, device, position, rotation, scale)
{
	this->speed = speed;
	this->distance = distance;
	originPos = myModel->position;
}

Bullet::Bullet(string name, string tag, Texture* textures[], unsigned int textureCount,
	TextureShader* shader, ID3D11Device* device, D3DXVECTOR3 position, D3DXVECTOR3 rotation, D3DXVECTOR3 scale, float speed, float distance)
		: GameObject(name, tag, textures, textureCount, shader, device, position, rotation, scale)
{
	this->speed = speed;
	this->distance = distance;
	originPos = myModel->position;
}


Bullet::~Bullet()
{
}

bool Bullet::Render(ID3D11DeviceContext* deviceContext, D3DXMATRIX worldMatrix, D3DXMATRIX viewMatrix, D3DXMATRIX projectionMatrix)
{
	UpdatePosition();

	bool result;
	myModel->Render(deviceContext);
	result = myShader->Render(deviceContext, myModel->GetIndexCount(), worldMatrix, viewMatrix, projectionMatrix, myTexture->GetTexture(), transparency);
	if (!result) return false;

	if (currentDistance >= distance)
	{
		System::checkGameObjects++;
		destroySignal = true;
	}

	return true;
}

void Bullet::UpdatePosition()
{
	myModel->position += (speed/200.0f*(D3DXVECTOR3(0.0f, 1.0f, 0.0f)));
	currentDistance = sqrt(pow(myModel->position.x - originPos.x, 2) + pow(myModel->position.y - originPos.y, 2));
}

float Bullet::GetSpeed()
{
	return speed;
}

float Bullet::GetDistance()
{
	return distance;
}
