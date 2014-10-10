#include "GameObject.h"


GameObject::GameObject()
{
	position = D3DXVECTOR3(0.0f, 0.0f, 0.0f);
	rotation = D3DXVECTOR3(0.0f, 0.0f, 0.0f);
	scale = D3DXVECTOR3(0.0f, 0.0f, 0.0f);
	myModel = nullptr;
	myTexture = nullptr;
	myShader = nullptr;
}

GameObject::GameObject(Model* model, Texture* texture, TextureShader* shader) : GameObject()
{
	myModel = model;
	myTexture = texture;
	myShader = shader;
}

GameObject::~GameObject()
{

}

bool GameObject::Render()
{

}

void GameObject::Destroy()
{

}
