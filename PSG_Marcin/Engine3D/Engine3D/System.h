#pragma once
// PREPROCESSOR
#include "Globals.h"

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
#include "Scene.h"

class GameObject;
class Graphics;
class Scene;

class System
{
private:
	LPCSTR applicationName;
	HINSTANCE m_hinstance;
	HWND m_hwnd;

	FPSCounter* m_FPS;
	CPUCounter* m_CPU;
	Timer* m_Timer;

	Scene* myScene;

	Input* myInput;
	Graphics* myGraphics;

	bool Frame();
	bool ProcessKeys();
	void ProcessCamera();
	void InitializeWindows(int&, int&);
	void ShutdownWindows();
	//void PlayerShoot();
	unsigned long keyTime;
public:
	static unsigned long frameCount;
	static bool playerAnimation;
	static float time;
	static unsigned long systemTime;
	static bool deferredFlag;

	System();
	System(const System&);
	~System();

	bool Initialize();
	void Shutdown();
	void Run();

	int GetCPUPercentage();
	int GetFPS();
	int GetTime();

	static void RotateVector(D3DXVECTOR3&, D3DXVECTOR3);

	LRESULT CALLBACK MessageHandler(HWND, UINT, WPARAM, LPARAM);
};

// FUNCTION PROTOTYPES
static LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);

// GLOBALS
static System* ApplicationHandle = 0;

/*
zamiast modyfikacji vertexów w buforze modyfikowaæ macierz
sortowanie obiektów ¿eby nie prze³¹czaæ tekstury
*/