#pragma once

#include <vector>
#include <random>
//#include <boost\property_tree\ptree.hpp>
//#include <boost\property_tree\xml_parser.hpp>
#include "Texture.h"
#include "GameObject.h"
#include "Direct3D.h"
#include "TextureManager.h"

class GameObject;

class Terrain
{
private:
	unsigned int width;
	unsigned int height;
	unsigned int borderWidth;
	float tileSize;
	float zPos;
	unsigned int textureCount;
	LightShader* terrainShader;
	Direct3D* myD3D;

	vector<Texture*> myTextures;
	vector<GameObject*> myTiles;

	void loadFromXML(string path, TextureManager* textureManager, Direct3D* myD3D);
public:
	Terrain();
	Terrain(unsigned int width, unsigned int height, unsigned int borderWidth, float tileSize, 
		float zPos, Texture* textures[], unsigned int textureCount, LightShader* terrainShader, Direct3D* myD3D);
	Terrain(string filePath, TextureManager* textureManager, LightShader* terrainShader, Direct3D* myD3D);
	Terrain(const Terrain&);
	~Terrain();

	bool Initialize();
	void Shutdown();

	vector<GameObject*> GetTiles();

	unsigned int GetWidth();
	unsigned int GetHeight();
	float GetSize();
};

