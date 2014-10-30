#include "Terrain.h"


Terrain::Terrain()
{
}

Terrain::Terrain(const Terrain& other)
{
}

Terrain::Terrain(unsigned int width, unsigned int height, float tileSize, float zPos, Texture* textures[], unsigned int textureCount, TextureShader* terrainShader, Direct3D* myD3D)
{
	this->width = width;
	this->height = height;
	this->tileSize = tileSize;
	this->zPos = zPos;
	this->textureCount = textureCount;
	this->terrainShader = terrainShader;
	this->myD3D = myD3D;

	for (int i = 0; i < textureCount; i++) myTextures.push_back(textures[i]);
}


Terrain::~Terrain()
{
}

bool Terrain::Initialize()
{
	srand(0);

	for (int i = 0; i < width; i++)
	{
		for (int j = 0; j < height; j++)
		{
			float xPos = ((float)i - (float)(width / 2))*tileSize*2;
			float yPos = ((float)j - (float)(height / 2))*tileSize*2;
			Texture* texture;
			if ((i == 0) || (i == width - 1) || (j == 0) || (j == height - 1))
			{
				texture = myTextures[textureCount - 1];
			}
			else
			{
				texture = myTextures[rand() % (textureCount - 1)];
			}

			GameObject* tile = new GameObject(
				"tile",
				"map_nocollid",
				texture,
				terrainShader,
				myD3D->GetDevice(),
				D3DXVECTOR3(xPos, yPos, zPos),
				D3DXVECTOR3(0.0f, 0.0f, 0.0f),
				D3DXVECTOR3(tileSize, tileSize, tileSize));
			myTiles.push_back(tile);
		}
	}

	return true;
}

void Terrain::Shutdown()
{
	for (vector<GameObject*>::iterator it = myTiles.begin(); it != myTiles.end(); ++it)
	{
		(*it)->Destroy();
		delete (*it);
	}
	myTiles.clear();
	myTextures.clear();
}

vector<GameObject*> Terrain::GetTiles()
{
	return myTiles;
}

unsigned int Terrain::GetWidth()
{
	return width;
}
unsigned int Terrain::GetHeight()
{
	return height;
}

float Terrain::GetSize()
{
	return tileSize;
}