#pragma once

#include <vector>
#include <random>
#include "Texture.h"
#include "GameObject.h"
#include "Direct3D.h"

class GameObject;

class Terrain
{
private:
	unsigned int width;
	unsigned int height;
	float tileSize;
	float zPos;
	unsigned int textureCount;
	TextureShader* terrainShader;
	Direct3D* myD3D;

	vector<Texture*> myTextures;
	vector<GameObject*> myTiles;
public:
	Terrain();
	Terrain(unsigned int width, unsigned int height, float tileSize, float zPos, Texture* textures[], unsigned int textureCount, TextureShader* terrainShader, Direct3D* myD3D);
	Terrain(const Terrain&);
	~Terrain();

	bool Initialize();
	void Shutdown();

	vector<GameObject*> GetTiles();

	unsigned int GetWidth();
	unsigned int GetHeight();
	float GetSize();
};

