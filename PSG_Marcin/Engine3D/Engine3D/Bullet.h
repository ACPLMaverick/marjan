#pragma once
#pragma message ("bullet")
#include "GameObject.h"

class GameObject;
class Scene;

class Bullet :
	public GameObject
{
protected:
	float speed;
	float distance;
	float currentDistance = 0.0f;
	D3DXVECTOR3 originPos;

	virtual void UpdatePosition();
public:
	Bullet();
	Bullet(const Bullet&);
	Bullet(string name, string tag, Texture* texture, TextureShader* shader, ID3D11Device* device, D3DXVECTOR3 position, D3DXVECTOR3 rotation, D3DXVECTOR3 scale, float speed, float distance);
	Bullet(string name, string tag, Texture* textures[], unsigned int textureCount,
		TextureShader* shader, ID3D11Device* device, D3DXVECTOR3 position, D3DXVECTOR3 rotation, D3DXVECTOR3 scale, float speed, float distance);
	~Bullet();

	virtual bool Render(ID3D11DeviceContext* deviceContext, D3DXMATRIX worldMatrix, D3DXMATRIX viewMatrix, D3DXMATRIX projectionMatrix);

	float GetSpeed();
	float GetDistance();
};

