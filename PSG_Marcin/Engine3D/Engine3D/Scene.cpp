#include "Scene.h"
unsigned int Scene::checkGameObjects;

Scene::Scene()
{
	checkGameObjects = false;
	player = nullptr;
}

Scene::Scene(const Scene &other)
{

}

Scene::~Scene()
{

}

void Scene::Initialize(string name, Graphics* myGraphics, HWND hwnd)
{
	this->myGraphics = myGraphics;
	this->m_hwnd = hwnd;
	this->name = name;

	InitializeGameObjects();
}

void Scene::Initialize(Graphics* myGraphics, HWND hwnd, string filePath)
{
	this->myGraphics = myGraphics;
	this->m_hwnd = hwnd;
	this->filePath = filePath;

	this->LoadFromFile();
	//InitializeGameObjects();
	//SaveToFile();
}

void Scene::LoadFromFile()
{
	ifstream is(filePath);
	is.open(filePath, ios_base::in);
	is.clear();

	string placek;
	is >> placek;

	if (placek != "Scene{") return;
	is >> placek;
	name = placek;

	while (!is.eof())
	{
		gameObjects.push_back(new GameObject(is, myGraphics, m_hwnd));
	}

	is.close();
}

void Scene::SaveToFile()
{
	ofstream of(filePath);
	of.clear();

	of << "Scene{\n";
	of << name + "\n";

	for (std::vector<GameObject*>::iterator it = gameObjects.begin(); it != gameObjects.end(); ++it)
	{
		(*it)->WriteToFile(of);
	}

	of.close();
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
}

void Scene::InitializeGameObjects()
{
	string texPath = "./Assets/Textures/noTexture.dds";
	string texPath02 = "./Assets/Textures/test.dds";
	string texPath03 = "./Assets/Textures/metal01_d.dds";
	GameObject* go01 = new GameObject(
		"player",
		"player",
		"./Assets/Models/DefaultSphere.obj",
		(myGraphics->GetTextures()->LoadTexture(myGraphics->GetD3D()->GetDevice(), texPath02.c_str())),
		(myGraphics->GetShaders())->LoadShader(myGraphics->GetD3D()->GetDevice(), m_hwnd, 0),
		myGraphics->GetD3D()->GetDevice(),
		D3DXVECTOR3(0.0f, 1.0f, 0.0f),
		D3DXVECTOR3(0.0f, 180.0f, 0.0f),
		D3DXVECTOR3(1.0f, 1.0f, 1.0f));
	player = go01;
	gameObjects.push_back(go01);

	GameObject* go02 = new GameObject(
		"plane",
		"terrain_collid",
		"./Assets/Models/BaseTerrain.obj",
		(myGraphics->GetTextures()->LoadTexture(myGraphics->GetD3D()->GetDevice(), texPath03.c_str())),
		(myGraphics->GetShaders())->LoadShader(myGraphics->GetD3D()->GetDevice(), m_hwnd, 0),
		myGraphics->GetD3D()->GetDevice(),
		D3DXVECTOR3(0.0f, 0.0f, 0.0f),
		D3DXVECTOR3(0.0f, 0.0f, 0.0f),
		D3DXVECTOR3(1.0f, 1.0f, 1.0f));
	gameObjects.push_back(go02);
}
//
//void Scene::InitializeTerrain()
//{
//	terrain = new Terrain("Configs/TerrainProperties.xml", myGraphics->GetTextures(), (myGraphics->GetShaders())->LoadShader(myGraphics->GetD3D()->GetDevice(), m_hwnd, 0), myGraphics->GetD3D());
//	terrain->Initialize();
//}

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
