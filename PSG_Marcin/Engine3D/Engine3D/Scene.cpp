#include "Scene.h"
unsigned int Scene::checkGameObjects;

Scene::Scene()
{
	checkGameObjects = false;
	player = nullptr;

	for (int i = 0; i < LIGHT_MAX_COUNT; i++)
	{
		lights[i] = nullptr;
	}
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

	if (TESTMODE)
	{
		InitializeLights();
		InitializeGameObjects();
	}
	else
	{
		this->LoadFromFile();
	}
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

	int i = 0;
	while (!is.eof())
	{
		is >> placek;
		if (placek == "GameObject{") gameObjects.push_back(new GameObject(is, myGraphics, m_hwnd));
		else if (placek == "LightAmbient{" && i < LIGHT_MAX_COUNT)
		{
			lights[i] = new LightAmbient(is);
			i++;
		}
		else if (placek == "LightDirectional{" && i < LIGHT_MAX_COUNT)
		{
			lights[i] = new LightDirectional(is);
			i++;
		}
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

	//for (int i = 0; i < LIGHT_MAX_COUNT; i++)
	//{
	//	if (lights[i] != nullptr) delete lights[i];
	//}
}

void Scene::InitializeGameObjects()
{
	int shaderCode = System::deferredFlag ? 3 : 2;
	string texPath = "./Assets/Textures/noTexture.dds";
	string texPath02 = "./Assets/Textures/test.dds";
	string texPath03 = "./Assets/Textures/metal01_d.dds";

	GameObject* ter = new GameObject(
		"plane",
		"terrain_collid",
		"./Assets/Models/BaseTerrain.obj",
		(myGraphics->GetTextures()->LoadTexture(myGraphics->GetD3D()->GetDevice(), texPath03.c_str())),
		(myGraphics->GetShaders())->LoadShader(myGraphics->GetD3D()->GetDevice(), m_hwnd, shaderCode),
		(DeferredShader*)(myGraphics->GetShaders())->LoadShader(myGraphics->GetD3D()->GetDevice(), m_hwnd, 3),
		myGraphics->GetD3D()->GetDevice(),
		D3DXVECTOR3(0.0f, 0.0f, 0.0f),
		D3DXVECTOR3(0.0f, 0.0f, 0.0f),
		D3DXVECTOR3(1.0f, 1.0f, 1.0f),
		D3DXVECTOR4(1.0f, 1.0f, 1.0f, 1.0f),
		1.0f,
		100.0f);
	gameObjects.push_back(ter);

	GameObject* go;
	float newx, newy, newz;
	int inlin = ((int)sqrt(OBJECTS_COUNT));
	int rozstrz = 4;

	for (int i = 0, j = 0, k = 0; i < OBJECTS_COUNT; i++)
	{
		newx = (float)(rozstrz * (j - inlin / 2));
		newy = 1.8f;
		newz = (float)(rozstrz * (k - inlin / 2));
		go = new GameObject(
			"ball",
			"ball_test",
			"./Assets/Models/DefaultSphere.obj",
			(myGraphics->GetTextures()->LoadTexture(myGraphics->GetD3D()->GetDevice(), texPath02.c_str())),
			(myGraphics->GetShaders())->LoadShader(myGraphics->GetD3D()->GetDevice(), m_hwnd, shaderCode),
			(DeferredShader*)(myGraphics->GetShaders())->LoadShader(myGraphics->GetD3D()->GetDevice(), m_hwnd, 3),
			myGraphics->GetD3D()->GetDevice(),
			D3DXVECTOR3(newx, newy, newz),
			D3DXVECTOR3(0.0f, 180.0f, 0.0f),
			D3DXVECTOR3(1.0f, 1.0f, 1.0f),
			D3DXVECTOR4(1.0f, 1.0f, 1.0f, 1.0f),
			1.0f,
			100.0f);
		gameObjects.push_back(go);
		if (i % inlin == 0 && i != 0)
		{
			j = 0;
			k++;
		}
		else j++;
	}
}
//
//void Scene::InitializeTerrain()
//{
//	terrain = new Terrain("Configs/TerrainProperties.xml", myGraphics->GetTextures(), (myGraphics->GetShaders())->LoadShader(myGraphics->GetD3D()->GetDevice(), m_hwnd, 0), myGraphics->GetD3D());
//	terrain->Initialize();
//}

void Scene::InitializeLights()
{
	srand(0);
	float random;
	lights[0] = new LightAmbient(D3DXVECTOR4(0.0f, 0.0f, 0.15f, 1.0f));
	for (int i = 1; i <= LIGHT_MAX_COUNT - 1; i++)
	{
		random = ((float)(rand() % 1001))/1000.0f;
		lights[i] = new LightDirectional(D3DXVECTOR4(random, random, random, 1.0f), D3DXVECTOR3(random, random, random));
	}
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

Light** Scene::GetLightArray()
{
	return lights;
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
