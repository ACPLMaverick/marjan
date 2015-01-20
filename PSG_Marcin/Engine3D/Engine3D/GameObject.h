#pragma once
#pragma message ("gameobject")
// includes
#include <d3d11.h>
#include <d3dx10math.h>
#include <vector>
//my classes
#include "System.h"
#include "Model.h"
#include "Sprite2D.h"
#include "Texture.h"
#include "TextureShader.h"
#include "SpecularShader.h"
#include "Model3D.h"
#include "Graphics.h"
//using namespace std;

class Graphics;

class GameObject
{
protected:
	bool destroySignal = false;
	string myName;
	string myTag;

	unsigned long animationLastFrame;
	unsigned int currentTextureID;
	vector<Texture*> animationTextures;
	bool canAnimate;

	float transparency = 1.0f;

	virtual bool InitializeModel(ID3D11Device* device, D3DXVECTOR3 position, D3DXVECTOR3 rotation, D3DXVECTOR3 scale);
	virtual bool InitializeModel(string modelPath, ID3D11Device* device, D3DXVECTOR3 position, D3DXVECTOR3 rotation, D3DXVECTOR3 scale);
	virtual void AnimateTexture();
public:
	Model* myModel;
	Texture* myTexture;
	TextureShader* myShader;

	D3DXVECTOR4 specularColor;
	float specularIntensity;
	float specularGlossiness;

	GameObject();
	GameObject(string name, string tag, Texture* texture, TextureShader* shader, ID3D11Device*, D3DXVECTOR3 position, D3DXVECTOR3 rotation, D3DXVECTOR3 scale);
	GameObject(string name, string tag, string modelPath, Texture* texture, TextureShader* shader, ID3D11Device*, D3DXVECTOR3 position, D3DXVECTOR3 rotation, D3DXVECTOR3 scale);
	GameObject(string name, string tag, string modelPath, Texture* texture, TextureShader* shader, ID3D11Device*, D3DXVECTOR3 position, D3DXVECTOR3 rotation, D3DXVECTOR3 scale,
		D3DXVECTOR4 specularColor, float specularIntensity, float specularGlossiness);
	GameObject(string name, string tag, Texture* animationTextures[], unsigned int textureCount, TextureShader* shader, ID3D11Device*, D3DXVECTOR3 position, D3DXVECTOR3 rotation, D3DXVECTOR3 scale);
	GameObject(ifstream &is, Graphics* myGraphics, HWND hwnd);
	GameObject(const GameObject&);
	~GameObject();

	virtual bool Render(ID3D11DeviceContext* deviceContext, D3DXMATRIX worldMatrix, D3DXMATRIX viewMatrix, D3DXMATRIX projectionMatrix, Light* lights[], D3DXVECTOR3 viewVector);
	void Destroy();

	bool GetDestroySignal();
	string GetName();
	string GetTag();

	void SetPosition(D3DXVECTOR3);
	void SetRotation(D3DXVECTOR3);
	void SetScale(D3DXVECTOR3);
	D3DXVECTOR3 GetPosition();
	D3DXVECTOR3 GetRotation();
	D3DXVECTOR3 GetScale();

	void SetTransparency(float);

	void WriteToFile(ofstream &of);
};

