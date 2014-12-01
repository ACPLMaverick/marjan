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
	string tmp = "";
	string tmp2;
	char tmpc;

	bool inGo = false;

	string goName;
	string goTag;
	string goTexture;
	string goShader;
	string goPos[3];
	string goRot[3];
	string goScale[3];

	Texture* newTex;
	TextureShader* newSh;
	GameObject* go01;

	goName.clear();
	goTag.clear();
	goTexture.clear();
	goShader.clear();
	for (int i = 0; i < 3; i++)
	{
		goPos[i].clear();
		goRot[i].clear();
		goScale[i].clear();
	}

	while (!is.eof())
	{
		tmp.clear();
		is >> tmp;
		tmp.push_back('\n');
		if (tmp.size() > 0)
		{
			tmp2.clear();
			int i = 0;
			for (; tmp.at(i) != '=' && tmp.at(i) != '\n' && tmp.at(i) != '\0'; i++)
			{
				tmpc = tmp.at(i);
				tmp2.push_back(tmpc);
			}
			i++;
			//tmp2.push_back('\0');
			//tmp2 = tmp2.substr(0, tmp2.size());
			if (!tmp2.compare("name") && !inGo)
			{
				tmp2.clear();
				for (; tmp.at(i) != '\n' && tmp.at(i) != '\0'; ++i)
				{
					tmpc = tmp.at(i);
					tmp2.push_back(tmpc);
				}
				this->name = tmp2;		// adding name
			}
			else if (!tmp2.compare("GameObject"))
			{
				inGo = true;
			}
			else if (!tmp2.compare("name") && inGo)
			{
				tmp2.clear();
				for (; tmp.at(i) != '\n'; i++)
				{
					tmpc = tmp.at(i);
					tmp2.push_back(tmpc);
				}
				goName = tmp2;		// adding name
			}
			else if (!tmp2.compare("tag") && inGo)
			{
				tmp2.clear();
				for (; tmp.at(i) != '\n'; i++)
				{
					tmpc = tmp.at(i);
					tmp2.push_back(tmpc);
				}
				goTag = tmp2;		// adding name
			}
			else if (!tmp2.compare("texture") && inGo)
			{
				tmp2.clear();
				for (; tmp.at(i) != '\n'; i++)
				{
					tmpc = tmp.at(i);
					tmp2.push_back(tmpc);
				}
				goTexture = tmp2;		// adding name
			}
			else if (!tmp2.compare("shader") && inGo)
			{
				tmp2.clear();
				for (; tmp.at(i) != '\n'; i++)
				{
					tmpc = tmp.at(i);
					tmp2.push_back(tmpc);
				}
				goShader = tmp2;		// adding name
			}
			else if (!tmp2.compare("position") && inGo)
			{
				tmp2.clear();
				for (int j = 0; tmp.at(i) != '\n'; i++)
				{
					tmpc = tmp.at(i);
					if (tmpc == ',' || tmpc == ']')
					{
						goPos[j] = tmp2;
						j++;
					}
					tmp2.push_back(tmpc);
					if (tmpc == '[' || tmpc == ',' || tmpc == ']') tmp2.clear();
				}
			}
			else if (!tmp2.compare("rotation") && inGo)
			{
				tmp2.clear();
				for (int j = 0; tmp.at(i) != '\n'; i++)
				{
					tmpc = tmp.at(i);
					if (tmpc == ',' || tmpc == ']')
					{
						goRot[j] = tmp2;
						j++;
					}
					tmp2.push_back(tmpc);
					if (tmpc == '[' || tmpc == ',' || tmpc == ']') tmp2.clear();
				}
			}
			else if (!tmp2.compare("scale") && inGo)
			{
				tmp2.clear();
				for (int j = 0; tmp.at(i) != '\n'; i++)
				{
					tmpc = tmp.at(i);
					if (tmpc == ',' || tmpc == ']')
					{
						goScale[j] = tmp2;
						j++;
					}
					tmp2.push_back(tmpc);
					if (tmpc == '[' || tmpc == ',' || tmpc == ']') tmp2.clear();
				}
			}
			else if (!tmp2.compare("}") && inGo)	// end of gameobject - create it
			{
				inGo = false;
				string japierdolechuj = goTexture;
				newTex = (myGraphics->GetTextures()->LoadTexture(myGraphics->GetD3D()->GetDevice(), japierdolechuj.c_str()));
				newSh = (myGraphics->GetShaders())->LoadShader(myGraphics->GetD3D()->GetDevice(), m_hwnd, stoi(goShader));
				D3DXVECTOR3 pos = D3DXVECTOR3(stof(goPos[0]), stof(goPos[1]), stof(goPos[2]));
				D3DXVECTOR3 rot = D3DXVECTOR3(stof(goRot[0]), stof(goRot[1]), stof(goRot[2]));
				D3DXVECTOR3 sc = D3DXVECTOR3(stof(goScale[0]), stof(goScale[1]), stof(goScale[2]));

				go01 = new GameObject(
					goName,
					goTag,
					newTex,
					newSh,
					myGraphics->GetD3D()->GetDevice(),
					pos,
					rot,
					sc);
				if(goTag == "player" && player == nullptr) player = go01;
				gameObjects.push_back(go01);
			}
		}
		}
		

		is.close();
	}

void Scene::SaveToFile()
{
	ofstream outputStream(filePath);
	outputStream << "name=" << name << endl;

	for (std::vector<GameObject*>::iterator it = gameObjects.begin(); it != gameObjects.end(); ++it)
	{
		outputStream << "GameObject=" << "{" << endl;

		outputStream << "name=" << (*it)->GetName() << endl;
		outputStream << "tag=" << (*it)->GetTag() << endl;
		outputStream << "texture=" << (*it)->myTexture->myName << endl;
		outputStream << "shader=" << (*it)->myShader->myID << endl;
		outputStream << "position=[" << (*it)->GetPosition().x << "," << (*it)->GetPosition().y << "," << (*it)->GetPosition().z << "]" << endl;
		outputStream << "rotation=[" << (*it)->GetRotation().x << "," << (*it)->GetRotation().y << "," << (*it)->GetRotation().z << "]" << endl;
		outputStream << "scale=[" << (*it)->GetScale().x << "," << (*it)->GetScale().y << "," << (*it)->GetScale().z << "]" << endl;

		outputStream << "}" << endl;
	}
	outputStream.close();
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
