#include "ShaderManager.h"


ShaderManager::ShaderManager()
{
}


ShaderManager::~ShaderManager()
{
	Shutdown();
}

TextureShader* ShaderManager::LoadShader(ID3D11Device* device, HWND hwnd, int id)
{
	// iterating over map to check for texture existence
	int i = 0;
	LPCSTR lastElem;
	for (map<LPCSTR, TextureShader*>::iterator it = shaders.begin(); it != shaders.end(); ++it, i++)
	{
		if (i == id) return (*it).second;
		lastElem = (*it).first;
	}

	if (shaders.empty()) return nullptr;

	// for safety returning last element of map
	return shaders[lastElem];
}

bool ShaderManager::AddShaders(ID3D11Device* device, HWND hwnd)
{
	// loading texture from drive and putting it into the map
	TextureShader* myShader = new TextureShader();
	bool result = myShader->Initialize(device, hwnd, 0);
	if (result) shaders.insert(pair<LPCSTR, TextureShader*>("TextureShader", myShader));
	else return false;

	LightShader* myLight = new LightShader();
	result = myLight->Initialize(device, hwnd, 1);
	if (result) shaders.insert(pair<LPCSTR, TextureShader*>("LightShader", myLight));
	else return false;

	SpecularShader* mySpec = new SpecularShader();
	result = mySpec->Initialize(device, hwnd, 2);
	if (result) shaders.insert(pair<LPCSTR, TextureShader*>("SpecularShader", mySpec));
	else return false;

	return result;
}

void ShaderManager::Shutdown()
{
	for (map<LPCSTR, TextureShader*>::iterator it = shaders.begin(); it != shaders.end(); ++it)
	{
		((*it).second)->Shutdown();
		delete ((*it).second);
		((*it).second) = nullptr;
	}
	shaders.clear();
}
