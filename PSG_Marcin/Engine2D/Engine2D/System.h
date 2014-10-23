#pragma once
// PREPROCESSOR
#define WIN32_LEAN_AND_MEAN

// INCLUDES
#include <Windows.h>
#include <vector>

// MY CLASS INCLUDES
#include "Graphics.h"
#include "Input.h"
#include "GameObject.h"
#include "ShaderManager.h"
#include "TextureManager.h"

class GameObject;
class Graphics;

class System
{
private:
	LPCSTR applicationName;
	HINSTANCE m_hinstance;
	HWND m_hwnd;

	Input* myInput;
	Graphics* myGraphics;

	vector<GameObject*> gameObjects;
	GameObject* player;

	bool Frame();
	bool ProcessKeys();
	void ProcessCamera();
	void InitializeWindows(int&, int&);
	void ShutdownWindows();
	void InitializeGameObjects();
	void CheckGameObjects();
	void PlayerShoot();
	GameObject* GetGameObjectByName(LPCSTR name);
	void GetGameObjectsByTag(LPCSTR tag, GameObject** ptr, unsigned int &count);
public:
	static unsigned int frameCount;
	static bool playerAnimation;
	static unsigned int checkGameObjects;

	System();
	System(const System&);
	~System();

	bool Initialize();
	void Shutdown();
	void Run();

	LRESULT CALLBACK MessageHandler(HWND, UINT, WPARAM, LPARAM);
};

// FUNCTION PROTOTYPES
static LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);

// GLOBALS
static System* ApplicationHandle = 0;