#include "GameObject.h"


GameObject::GameObject()
{
	myModel = nullptr;
	myTexture = nullptr;
	myShader = nullptr;
}

GameObject::GameObject(string name, string tag, Texture* texture, TextureShader* shader, ID3D11Device* device, D3DXVECTOR3 position, D3DXVECTOR3 rotation, D3DXVECTOR3 scale) : GameObject()
{
	myName = name;
	myTag = tag;

	myTexture = texture;
	myShader = shader;
	InitializeModel(device, position, rotation, scale);
	animationLastFrame = System::frameCount;
}

GameObject::GameObject(string name, string tag, Texture* textures[], unsigned int textureCount, 
	TextureShader* shader, ID3D11Device* device, D3DXVECTOR3 position, D3DXVECTOR3 rotation, D3DXVECTOR3 scale) : GameObject()
{
	myName = name;
	myTag = tag;

	for (int i = 0; i < textureCount; i++) animationTextures.push_back(textures[i]);
	currentTextureID = 0;
	myTexture = animationTextures.at(currentTextureID);

	myShader = shader;
	InitializeModel(device, position, rotation, scale);
}

GameObject::~GameObject()
{
	//Destroy();
	// possible memory leak?
}

bool GameObject::InitializeModel(ID3D11Device* device, D3DXVECTOR3 position, D3DXVECTOR3 rotation, D3DXVECTOR3 scale)
{
	bool result;
	myModel = new Sprite2D(position, rotation, scale);
	if (!myModel) return false;
	result = myModel->Initialize(device, myTexture);
	if (!result) return false;
}

bool GameObject::Render(ID3D11DeviceContext* deviceContext, D3DXMATRIX worldMatrix, D3DXMATRIX viewMatrix, D3DXMATRIX projectionMatrix)
{
	if(animationTextures.size() > 0)AnimateTexture();
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

void GameObject::AnimateTexture()
{
	if (System::frameCount - animationLastFrame >= 600)
	{
		animationLastFrame = System::frameCount;
		currentTextureID = (currentTextureID + 1) % animationTextures.size();
		myTexture = animationTextures[currentTextureID];
	}
}

string GameObject::GetName()
{
	return myName;
}

string GameObject::GetTag()
{
	return myTag;
}

void GameObject::SetPosition(D3DXVECTOR3 position)
{
	myModel->position = position;
}

void GameObject::SetRotation(D3DXVECTOR3 rotation)
{
	myModel->rotation = rotation;
}


void GameObject::SetScale(D3DXVECTOR3 scale)
{
	myModel->scale = scale;
}


D3DXVECTOR3 GameObject::GetPosition()
{
	return myModel->position;
}


D3DXVECTOR3 GameObject::GetRotation()
{
	return myModel->rotation;
}


D3DXVECTOR3 GameObject::GetScale()
{
	return myModel->scale;
}

