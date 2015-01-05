#include "Bullet.h"


Bullet::Bullet() : GameObject()
{
}

Bullet::Bullet(const Bullet &other)
{
}

Bullet::Bullet(string name, string tag, Texture* texture, LightShader* shader, ID3D11Device* device, D3DXVECTOR3 position, D3DXVECTOR3 rotation, D3DXVECTOR3 scale, float speed, float distance)
	: GameObject(name, tag, texture, shader, device, position, rotation, scale)
{
	this->speed = speed;
	this->distance = distance;
	originPos = myModel->position;
}

Bullet::Bullet(string name, string tag, Texture* textures[], unsigned int textureCount,
	LightShader* shader, ID3D11Device* device, D3DXVECTOR3 position, D3DXVECTOR3 rotation, D3DXVECTOR3 scale, float speed, float distance)
		: GameObject(name, tag, textures, textureCount, shader, device, position, rotation, scale)
{
	this->speed = speed;
	this->distance = distance;
	originPos = myModel->position;
}


Bullet::~Bullet()
{
}

bool Bullet::Render(ID3D11DeviceContext* deviceContext, D3DXMATRIX worldMatrix, D3DXMATRIX viewMatrix, D3DXMATRIX projectionMatrix, LightDirectional* light)
{
	UpdatePosition();

	bool result;
	myModel->Render(deviceContext);
	result = myShader->Render(deviceContext, myModel->GetIndexCount(), worldMatrix, viewMatrix, projectionMatrix, myTexture->GetTexture(), light->GetDiffuseColor(), light->GetDirection());
	if (!result) return false;

	if (currentDistance >= distance)
	{
		Scene::checkGameObjects++;
		destroySignal = true;
	}

	return true;
}

void Bullet::UpdatePosition()
{
	D3DXVECTOR3 rotationVector(0.0f, 1.0f, 0.0f);

	// rotation
	D3DXMATRIX rotateX;
	D3DXMATRIX rotateY;
	D3DXMATRIX rotateZ;
	D3DXMatrixRotationX(&rotateX, myModel->rotation.x);
	D3DXMatrixRotationY(&rotateY, myModel->rotation.y);
	D3DXMatrixRotationZ(&rotateZ, myModel->rotation.z);
	D3DXMATRIX rotationMatrix = rotateX*rotateY*rotateZ;
	D3DXVECTOR4 outputVec;

	D3DXVec3Transform(&outputVec, &rotationVector, &rotationMatrix);
	rotationVector.x = outputVec.x;
	rotationVector.y = outputVec.y;
	rotationVector.z = outputVec.z;

	myModel->position += (speed/200.0f*rotationVector*System::time);
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
