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

GameObject::GameObject(string name, string tag, Texture* texture, TextureShader* shader, ID3D11Device* device) : GameObject()
{
	myName = name;
	myTag = tag;
	
	myTexture = texture;
	InitializeModel(device);
	myShader = shader;
	position = myModel->position;
	rotation = myModel->rotation;
	scale = myModel->scale;
}

GameObject::~GameObject()
{

}

bool GameObject::InitializeModel(ID3D11Device* device)
{
	bool result;
	myModel = new Sprite2D(D3DXVECTOR3(0.0f, 0.0f, 0.0f), D3DXVECTOR3(0.0f, 0.0f, 0.0f), D3DXVECTOR3(1.0f, 1.0f, 1.0f), nullptr);
	if (!myModel) return false;
	result = myModel->Initialize(device, "./Assets/Textures/noTexture.dds");
	if (!result) return false;
}

bool GameObject::Render(ID3D11DeviceContext* deviceContext, D3DXMATRIX worldMatrix, D3DXMATRIX viewMatrix, D3DXMATRIX projectionMatrix)
{
	bool result;
	myModel->Render(deviceContext);
	result = myShader->Render(deviceContext, myModel->GetIndexCount(), worldMatrix, viewMatrix, projectionMatrix, myTexture->GetTexture());
	if (!result) return false;
	return true;
}

void GameObject::Destroy()
{
	myModel->Shutdown();
	myModel = nullptr;
}

string GameObject::GetName()
{
	return myName;
}

string GameObject::GetTag()
{
	return myTag;
}