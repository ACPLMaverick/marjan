#include "Terrain.h"


Terrain::Terrain()
{
}

Terrain::Terrain(const Terrain& other)
{
}

Terrain::Terrain(unsigned int width, unsigned int height, unsigned int borderWidth, float tileSize, float zPos, 
	Texture* textures[], unsigned int textureCount, TextureShader* terrainShader, Direct3D* myD3D)
{
	this->width = width;
	this->height = height;
	this->borderWidth = borderWidth;
	this->tileSize = tileSize;
	this->zPos = zPos;
	this->textureCount = textureCount;
	this->terrainShader = terrainShader;
	this->myD3D = myD3D;

	for (int i = 0; i < textureCount; i++) myTextures.push_back(textures[i]);
}

Terrain::Terrain(string filePath, TextureManager* textureManager, TextureShader* terrainShader, Direct3D* myD3D)
{
	this->textureCount = textureCount;
	this->terrainShader = terrainShader;
	this->myD3D = myD3D;

	//for (int i = 0; i < textureCount; i++) myTextures.push_back(textures[i]);

	loadFromXML(filePath, textureManager, myD3D);
}


Terrain::~Terrain()
{
}

bool Terrain::Initialize()
{
	srand(0);

	for (unsigned int i = 0; i < width; i++)
	{
		for (unsigned int j = 0; j < height; j++)
		{
			float xPos = ((float)i - (float)(width / 2))*tileSize*2;
			float yPos = ((float)j - (float)(height / 2))*tileSize*2;
			Texture* texture;
			if ((i >= 0 && i < borderWidth) || (i > (width - 1 - borderWidth)) || 
				(j >= 0 && j < borderWidth) || (j > (height - 1 - borderWidth)))
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

void Terrain::loadFromXML(string path, TextureManager* textureManager, Direct3D* myD3D)
{
	using boost::property_tree::ptree;
	using boost::property_tree::read_xml;

	string strWidth;
	string strHeight;
	string strBorderWidth;
	string strTileSize;
	string strZPos;
	vector<string> tilePaths;
	string pathPrefix = "./Assets/Textures/";

	ptree pt;
	read_xml(path, pt);

	strWidth = pt.get<string>("TerrainProperties.Width");
	strHeight = pt.get<string>("TerrainProperties.Height");
	strBorderWidth = pt.get<string>("TerrainProperties.BorderWidth");
	strTileSize = pt.get<string>("TerrainProperties.TileSize");
	strZPos = pt.get<string>("TerrainProperties.ZPos");

	ptree children = pt.get_child("TerrainProperties.TerrainTiles");
	int tileCount = children.count("TilePath");
	this->textureCount = tileCount;

	for (int i = 0; i < tileCount; i++)
	{
		children.pop_front();
		tilePaths.push_back(pathPrefix + children.get<string>("TilePath"));
	}

	this->width = stoi(strWidth);
	this->height = stoi(strHeight);
	this->borderWidth = stoi(strBorderWidth);
	this->tileSize = stof(strTileSize);
	this->zPos = stof(strZPos);

	for (vector<string>::iterator it = tilePaths.begin(); it != tilePaths.end(); ++it)
	{
		myTextures.push_back(textureManager->LoadTexture(myD3D->GetDevice(), (*it).c_str()));
	}
}