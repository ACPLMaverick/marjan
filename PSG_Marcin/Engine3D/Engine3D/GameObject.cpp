#include "GameObject.h"


GameObject::GameObject()
{
	myModel = nullptr;
	myTexture = nullptr;
	myShader = nullptr;

	specularColor = D3DXVECTOR4(1.0f, 1.0f, 1.0f, 1.0f);
	specularIntensity = 1.0f;
	specularGlossiness = 100.0f;
}

GameObject::GameObject(string name, string tag, Texture* texture, TextureShader* shader, DeferredShader* deferredShader, ID3D11Device* device, D3DXVECTOR3 position, D3DXVECTOR3 rotation, D3DXVECTOR3 scale) : GameObject()
{
	myName = name;
	myTag = tag;

	myTexture = texture;
	myShader = shader;
	this->deferredShader = deferredShader;
	InitializeModel(device, position, rotation, scale);
	animationLastFrame = System::frameCount;
}

GameObject::GameObject(string name, string tag, string modelPath, Texture* texture, TextureShader* shader, DeferredShader* deferredShader, ID3D11Device* device, D3DXVECTOR3 position, D3DXVECTOR3 rotation, D3DXVECTOR3 scale) : GameObject()
{
	myName = name;
	myTag = tag;

	myTexture = texture;
	myShader = shader;
	this->deferredShader = deferredShader;
	InitializeModel(modelPath, device, position, rotation, scale);
	animationLastFrame = System::frameCount;
}

GameObject::GameObject(string name, string tag, string modelPath, Texture* texture, TextureShader* shader, DeferredShader* deferredShader, ID3D11Device* device, D3DXVECTOR3 position, D3DXVECTOR3 rotation, D3DXVECTOR3 scale,
	D3DXVECTOR4 specularColor, float specularIntensity, float specularGlossiness) : GameObject(name, tag, modelPath, texture, shader, deferredShader, device, position, rotation, scale)
{
	this->specularColor = specularColor;
	this->specularIntensity = specularIntensity;
	this->specularGlossiness = specularGlossiness;
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

GameObject::GameObject(ifstream &is, Graphics* myGraphics, HWND hwnd) : GameObject()
{
	string line;
	string modelPath;
	string texturePath;
	int shaderID;
	D3DXVECTOR3 pos;
	D3DXVECTOR3 rot;
	D3DXVECTOR3 scl;

	//is >> line;

	//if (line != "GameObject{") GameObject();

	is >> myName;
	is >> myTag;
	is >> modelPath;
	is >> texturePath;
	is >> shaderID;

	is >> specularColor.x;
	is >> specularColor.y;
	is >> specularColor.z;
	is >> specularColor.w;
	is >> specularIntensity;
	is >> specularGlossiness;

	is >> pos.x;
	is >> pos.y;
	is >> pos.z;
	is >> rot.x;
	is >> rot.y;
	is >> rot.z;
	is >> scl.x; 
	is >> scl.y;
	is >> scl.z;

	is >> line;

	myTexture = myGraphics->GetTextures()->LoadTexture(myGraphics->GetD3D()->GetDevice(), texturePath);
	myShader = myGraphics->GetShaders()->LoadShader(myGraphics->GetD3D()->GetDevice(), hwnd, shaderID);
	deferredShader = (DeferredShader*) myGraphics->GetShaders()->LoadShader(myGraphics->GetD3D()->GetDevice(), hwnd, 3);

	InitializeModel(modelPath, myGraphics->GetD3D()->GetDevice(), pos, rot, scl);
}

GameObject::~GameObject()
{
	//Destroy();
	// possible memory leak?
}

bool GameObject::Render(ID3D11DeviceContext* deviceContext, D3DXMATRIX worldMatrix, D3DXMATRIX viewMatrix, D3DXMATRIX projectionMatrix, Light* lights[], D3DXVECTOR3 viewVector)
{
	if (myTag == "player") canAnimate = System::playerAnimation;
	if (animationTextures.size() > 0 && canAnimate)AnimateTexture();
	bool result;
	myModel->Render(deviceContext);

	int count = 0;
	D3DXVECTOR4 cols[LIGHT_MAX_COUNT];
	D3DXVECTOR4 dirs[LIGHT_MAX_COUNT];
	LightAmbient* ambient;

	if (lights != NULL)
	{
		ambient = (LightAmbient*)lights[0];

		for (; lights[count + 1] != nullptr && count <= LIGHT_MAX_COUNT; count++);

		for (int i = 1; i <= count && i < LIGHT_MAX_COUNT; i++)
		{
			cols[i - 1] = ((LightDirectional*)lights[i])->GetDiffuseColor();
			dirs[i - 1] = D3DXVECTOR4(((LightDirectional*)lights[i])->GetDirection().x, ((LightDirectional*)lights[i])->GetDirection().y,
				((LightDirectional*)lights[i])->GetDirection().z, 0);
		}
		cols[0].w = count;
	}
	
	if (!System::deferredFlag)
	{
		if (myShader->myID == 0)
		{
			result = myShader->Render(deviceContext, myModel->GetIndexCount(), worldMatrix, viewMatrix, projectionMatrix, myTexture->GetTexture(), 1.0f);
		}
		else if (myShader->myID == 1)
		{
			LightShader* ls = (LightShader*)myShader;
			result = ls->Render(deviceContext, myModel->GetIndexCount(), worldMatrix, viewMatrix, projectionMatrix, myTexture->GetTexture(), cols, dirs, count, ambient->GetDiffuseColor());
		}
		else if (myShader->myID == 2)
		{
			SpecularShader* ls = (SpecularShader*)myShader;
			result = ls->Render(deviceContext, myModel->GetIndexCount(), worldMatrix, viewMatrix, projectionMatrix, myTexture->GetTexture(),
				cols, dirs, count, ambient->GetDiffuseColor(), viewVector, this->specularColor, this->specularIntensity, this->specularGlossiness);
		}
	}
	else 
	{
		result = deferredShader->Render(deviceContext, myModel->GetIndexCount(), worldMatrix, viewMatrix, projectionMatrix, myTexture->GetTexture(), 1.0f);
	}
	
	if (!result) return false;
	return true;
}

bool GameObject::InitializeModel(ID3D11Device* device, D3DXVECTOR3 position, D3DXVECTOR3 rotation, D3DXVECTOR3 scale)
{
	if (myTag == "player") canAnimate = true;
	else canAnimate = false;
	bool result;

	D3D11_USAGE usage;
	if (myTag == "terrain_nocollid" || myTag == "terrain_collid") usage = D3D11_USAGE_IMMUTABLE;
	else usage = D3D11_USAGE_DYNAMIC;
	myModel = new Model3D(position, rotation, scale, usage, "");
	if (!myModel) return false;
	result = myModel->Initialize(device, myTexture);
	if (!result) return false;
}

bool GameObject::InitializeModel(string modelPath, ID3D11Device* device, D3DXVECTOR3 position, D3DXVECTOR3 rotation, D3DXVECTOR3 scale)
{
	if (myTag == "player") canAnimate = true;
	else canAnimate = false;
	bool result;

	D3D11_USAGE usage;
	if (myTag == "terrain_nocollid" || myTag == "terrain_collid") usage = D3D11_USAGE_IMMUTABLE;
	else usage = D3D11_USAGE_DYNAMIC;
	myModel = new Model3D(position, rotation, scale, usage, modelPath);
	if (!myModel) return false;
	result = myModel->Initialize(device, myTexture);
	if (!result) return false;
}

void GameObject::Destroy()
{
	myModel->Shutdown();
	myModel = nullptr;
}

void GameObject::AnimateTexture()
{
	if ((System::frameCount - animationLastFrame) > (int)(20/(System::time != 0 ? System::time : 1)))
	{
		animationLastFrame = System::frameCount;
		currentTextureID = (currentTextureID + 1 /*+ (int)System::time*/) % animationTextures.size();
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

void GameObject::SetTransparency(float tr)
{
	transparency = tr;
}

bool GameObject::GetDestroySignal()
{
	return destroySignal;
}

void GameObject::WriteToFile(ofstream &of)
{
	of << "GameObject{\n";
	of << myName + "\n";
	of << myTag + "\n";
	of << myModel->GetFilePath() + "\n";
	of << myTexture->myName + "\n";
	of << myShader->myID + "\n";
	of << to_string(specularColor.x) + " " + to_string(specularColor.y) + " " + to_string(specularColor.z) + " " + to_string(specularColor.w) + "\n";
	of << to_string(specularIntensity) + "\n";
	of << to_string(specularGlossiness) + "\n";
	of << to_string(myModel->position.x) + " " + to_string(myModel->position.y) + " " + to_string(myModel->position.z) + "\n";
	of << to_string(myModel->rotation.x) + " " + to_string(myModel->rotation.y) + " " + to_string(myModel->rotation.z) + "\n";
	of << to_string(myModel->scale.x) + " " + to_string(myModel->scale.y) + " " + to_string(myModel->scale.z) + "\n";
	of << "}\n";
}
