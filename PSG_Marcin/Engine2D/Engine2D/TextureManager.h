#pragma once
#include <map>
#include <D3D11.h>
#include "Texture.h"
using namespace std;

class TextureManager
{
private:
	map<LPCSTR, Texture*> textures;
public:
	TextureManager();
	~TextureManager();

	Texture* LoadTexture(ID3D11Device* device, LPCSTR path);
	Texture* LoadTexture(ID3D11Device* device, int id);
	bool AddTexture(ID3D11Device* device, LPCSTR path);
	void Shutdown();
};

