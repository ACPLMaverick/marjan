#pragma once

#include <vector>
#include <Windows.h>

#include "Graphics.h"
#include "GameObject.h"
#include "Terrain.h"

class GameObject;
class Graphics;
class Terrain;

class Scene
{
private:
	string name;
	
	Graphics* myGraphics;
	HWND m_hwnd;

	vector<GameObject*> gameObjects;
	Terrain* terrain;
	GameObject* player;

	void InitializeGameObjects();
	void InitializeTerrain();
public:
	static unsigned int checkGameObjects;

	Scene();
	Scene(const Scene&);
	~Scene();

	void Initialize(Graphics* myGraphics, HWND hwnd);
	void Shutdown();

	void Add(GameObject* obj);
	void Remove(unsigned int number);
	void Remove(string name);
	void Remove(string tag, int count);

	void CheckGameObjects();

	GameObject* GetPlayer();
	GameObject** GetGameObjectsAsArray();
	unsigned int GetGameObjectsSize();
	GameObject* GetGameObjectByName(LPCSTR name);
	void GetGameObjectsByTag(LPCSTR tag, GameObject** ptr, unsigned int &count);
};

