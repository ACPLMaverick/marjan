#include "Scene.h"
unsigned int Scene::checkGameObjects;

Scene::Scene()
{
	checkGameObjects = false;
	terrain = nullptr;
	player = nullptr;
}

Scene::Scene(const Scene &other)
{

}

Scene::~Scene()
{

}

void Scene::Initialize(Graphics* myGraphics, HWND hwnd)
{
	this->myGraphics = myGraphics;
	this->m_hwnd = hwnd;

	InitializeGameObjects();
}

void Scene::Shutdown()
{
	for (std::vector<GameObject*>::iterator it = gameObjects.begin(); it != gameObjects.end(); ++it)
	{
		if (*it) (*it)->Destroy();
		delete (*it);
		(*it) = nullptr;
	}
	gameObjects.clear();

	if (terrain)
	{
		terrain->Shutdown();
		delete terrain;
		terrain = nullptr;
	}
}

void Scene::InitializeGameObjects()
{
	string texPath = "./Assets/Textures/noTexture.dds";
	string texPath02 = "./Assets/Textures/test.dds";
	string texPath03 = "./Assets/Textures/metal01_d.dds";
	GameObject* go01 = new GameObject(
		"player",
		"player",
		(myGraphics->GetTextures()->LoadTexture(myGraphics->GetD3D()->GetDevice(), texPath.c_str())),
		(myGraphics->GetShaders())->LoadShader(myGraphics->GetD3D()->GetDevice(), m_hwnd, 0),
		myGraphics->GetD3D()->GetDevice(),
		D3DXVECTOR3(0.0f, 0.9f, 0.0f),
		D3DXVECTOR3(0.0f, 0.0f, 0.0f),
		D3DXVECTOR3(1.0f, 1.0f, 1.0f));
	player = go01;
	gameObjects.push_back(go01);
	GameObject* go02 = new GameObject(
		"dupa",
		"dupa",
		(myGraphics->GetTextures()->LoadTexture(myGraphics->GetD3D()->GetDevice(), texPath02.c_str())),
		(myGraphics->GetShaders())->LoadShader(myGraphics->GetD3D()->GetDevice(), m_hwnd, 0),
		myGraphics->GetD3D()->GetDevice(),
		D3DXVECTOR3(4.0f, 2.0f, 4.0f),
		D3DXVECTOR3(0.0f, 0.0f, 0.0f),
		D3DXVECTOR3(1.0f, 2.0f, 1.0f));
	gameObjects.push_back(go02);
	GameObject* go03 = new GameObject(
		"terrain",
		"terrain_collid",
		(myGraphics->GetTextures()->LoadTexture(myGraphics->GetD3D()->GetDevice(), texPath03.c_str())),
		(myGraphics->GetShaders())->LoadShader(myGraphics->GetD3D()->GetDevice(), m_hwnd, 0),
		myGraphics->GetD3D()->GetDevice(),
		D3DXVECTOR3(0.0f, -30.0f, 0.0f),
		D3DXVECTOR3(0.0f, 0.0f, 0.0f),
		D3DXVECTOR3(30.0f, 30.0f, 30.0f));
	gameObjects.push_back(go03);
}

void Scene::InitializeTerrain()
{
	terrain = new Terrain("Configs/TerrainProperties.xml", myGraphics->GetTextures(), (myGraphics->GetShaders())->LoadShader(myGraphics->GetD3D()->GetDevice(), m_hwnd, 0), myGraphics->GetD3D());
	terrain->Initialize();
}

void Scene::Add(GameObject* obj)
{
	gameObjects.push_back(obj);
	if (obj->GetName() == "player" && player == nullptr) player = obj;
}

void Scene::Remove(unsigned int number)
{
	// TODO: IMPLEMENT
}

void Scene::Remove(string name)
{
	// TODO: IMPLEMENT
}

void Scene::Remove(string tag, int count)
{
	// TODO: IMPLEMENT
}

void Scene::CheckGameObjects()
{
	while (checkGameObjects > 0)
	{
		for (std::vector<GameObject*>::iterator it = gameObjects.begin(); it != gameObjects.end(); ++it)
		{
			if ((*it)->GetDestroySignal())
			{
				(*it)->Destroy();
				delete (*it);
				(*it) = nullptr;
				gameObjects.erase(it);
				break;
			}
		}
		checkGameObjects--;
	}
}

GameObject* Scene::GetPlayer()
{
	return player;
}

GameObject* Scene::GetGameObjectByName(LPCSTR name)
{
	for (std::vector<GameObject*>::iterator it = gameObjects.begin(); it != gameObjects.end(); ++it)
	{
		if ((*it)->GetName() == name) return (*it);
	}
	return nullptr;
}

void Scene::GetGameObjectsByTag(LPCSTR tag, GameObject** ptr, unsigned int &count)
{
	unsigned int c = 0;
	for (std::vector<GameObject*>::iterator it = gameObjects.begin(); it != gameObjects.end(); ++it)
	{
		if ((*it)->GetTag() == tag) c++;
	}
	ptr = new GameObject*[c];
	c = 0;
	for (std::vector<GameObject*>::iterator it = gameObjects.begin(); it != gameObjects.end(); ++it)
	{
		if ((*it)->GetTag() == tag)
		{
			ptr[c] = (*it);
			c++;
		}
	}
	count = c;
}

GameObject** Scene::GetGameObjectsAsArray()
{
	GameObject** goTab = new GameObject*[gameObjects.size()];
	for (int i = 0; i < gameObjects.size(); i++)
	{
		goTab[i] = gameObjects.at(i);
	}
	return goTab;
}

unsigned int Scene::GetGameObjectsSize()
{
	return gameObjects.size();
}
