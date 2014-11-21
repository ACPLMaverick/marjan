#include "TextureManager.h"


TextureManager::TextureManager()
{
}


TextureManager::~TextureManager()
{
	Shutdown();
}

Texture* TextureManager::LoadTexture(ID3D11Device* device, LPCSTR path)
{
	// iterating over map to check for texture existence
	for (map<LPCSTR, Texture*>::iterator it = textures.begin(); it != textures.end(); ++it)
	{
		if ((*it).first == path) return (*it).second;
	}

	// loading texture from drive and putting it into the map
	Texture* myTexture = new Texture();
	myTexture->Initialize(device, path);
	textures.insert(pair<LPCSTR, Texture*>(path, myTexture));
	return myTexture;
}

Texture* TextureManager::LoadTexture(ID3D11Device* device, int id)
{
	// iterating over map to check for texture existence
	int i = 0;
	LPCSTR lastElem;
	for (map<LPCSTR, Texture*>::iterator it = textures.begin(); it != textures.end(); ++it, i++)
	{
		if (i == id) return (*it).second;
		lastElem = (*it).first;
	}

	if (textures.empty()) return nullptr;

	// for safety returning last element of map
	return textures[lastElem];
}

bool TextureManager::AddTexture(ID3D11Device* device, LPCSTR path)
{
	// loading texture from drive and putting it into the map
	Texture* myTexture = new Texture();
	bool result = myTexture->Initialize(device, path);
	if(result) textures.insert(pair<LPCSTR, Texture*>(path, myTexture));
	return result;
}

void TextureManager::Shutdown()
{
	for (map<LPCSTR, Texture*>::iterator it = textures.begin(); it != textures.end(); ++it)
	{
		((*it).second)->Shutdown();
		delete ((*it).second);
		((*it).second) = nullptr;
	}
	textures.clear();
}