#pragma once

// includes
#include <d3d11.h>
#include <d3dx10math.h>
//my classes
#include "Model.h"
#include "Texture.h"
#include "TextureShader.h"

class GameObject
{
protected:
	Model* myModel;
	Texture* myTexture;
	TextureShader* myShader;
public:
	D3DXVECTOR3 position;
	D3DXVECTOR3 rotation;
	D3DXVECTOR3 scale;

	GameObject();
	GameObject(Model* model, Texture* texture, TextureShader* shader);
	GameObject(const GameObject&);
	~GameObject();

	bool Render();
	void Destroy();
};

