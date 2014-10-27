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
#include "FPSCounter.h"
#include "CPUCounter.h"
#include "Timer.h"
#include "Terrain.h"

class GameObject;
class Graphics;
class Terrain;

class System
{
private:
	LPCSTR applicationName;
	HINSTANCE m_hinstance;
	HWND m_hwnd;

	FPSCounter* m_FPS;
	CPUCounter* m_CPU;
	Timer* m_Timer;

	Input* myInput;
	Graphics* myGraphics;

	vector<GameObject*> gameObjects;
	Terrain* terrain;
	GameObject* player;

	bool Frame();
	bool ProcessKeys();
	void ProcessCamera();
	void InitializeWindows(int&, int&);
	void ShutdownWindows();
	void InitializeGameObjects();
	void InitializeTerrain();
	void CheckGameObjects();
	void PlayerShoot();
	GameObject* GetGameObjectByName(LPCSTR name);
	void GetGameObjectsByTag(LPCSTR tag, GameObject** ptr, unsigned int &count);
public:
	static unsigned long frameCount;
	static bool playerAnimation;
	static unsigned int checkGameObjects;
	static float time;
	static unsigned long systemTime;

	System();
	System(const System&);
	~System();

	bool Initialize();
	void Shutdown();
	void Run();

	int GetCPUPercentage();
	int GetFPS();
	int GetTime();

	LRESULT CALLBACK MessageHandler(HWND, UINT, WPARAM, LPARAM);
};

// FUNCTION PROTOTYPES
static LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);

// GLOBALS
static System* ApplicationHandle = 0;