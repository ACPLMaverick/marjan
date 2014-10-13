#pragma once

// includes
#include <d3d11.h>
#include <d3dx10math.h>
//my classes
#include "Model.h"
#include "Sprite2D.h"
#include "Texture.h"
#include "TextureShader.h"
using namespace std;

class GameObject
{
protected:
	string myName;
	string myTag;

	Model* myModel;
	Texture* myTexture;
	TextureShader* myShader;

	virtual bool InitializeModel(ID3D11Device* device);
public:
	D3DXVECTOR3 position;
	D3DXVECTOR3 rotation;
	D3DXVECTOR3 scale;

	GameObject();
	GameObject(string name, string tag, Texture* texture, TextureShader* shader, ID3D11Device*, D3DXVECTOR3 position, D3DXVECTOR3 rotation, D3DXVECTOR3 scale);
	GameObject(const GameObject&);
	~GameObject();

	bool Render(ID3D11DeviceContext* deviceContext, D3DXMATRIX worldMatrix, D3DXMATRIX viewMatrix, D3DXMATRIX projectionMatrix);
	void Destroy();

	string GetName();
	string GetTag();
};

