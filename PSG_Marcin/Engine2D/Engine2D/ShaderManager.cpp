#include "ShaderManager.h"


ShaderManager::ShaderManager()
{
}


ShaderManager::~ShaderManager()
{
	Shutdown();
}

TextureShader* ShaderManager::LoadShader(ID3D11Device* device, HWND hwnd, LPCSTR name)
{
	// iterating over map to check for texture existence
	for (map<LPCSTR, TextureShader*>::iterator it = shaders.begin(); it != shaders.end(); ++it)
	{
		if ((*it).first == name) return (*it).second;
	}

	// loading texture from drive and putting it into the map
	TextureShader* myShader = new TextureShader();
	myShader->Initialize(device, hwnd);
	shaders.insert(pair<LPCSTR, TextureShader*>(name, myShader));
	return myShader;
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

bool ShaderManager::AddShader(ID3D11Device* device, HWND hwnd, LPCSTR name)
{
	// loading texture from drive and putting it into the map
	TextureShader* myShader = new TextureShader();
	bool result = myShader->Initialize(device, hwnd);
	if (result) shaders.insert(pair<LPCSTR, TextureShader*>(name, myShader));
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
