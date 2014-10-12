#ifndef _SYSTEM_H_
#define _SYSTEM_H_

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

class System
{
private:
	LPCSTR applicationName;
	HINSTANCE m_hinstance;
	HWND m_hwnd;

	Input* myInput;
	Graphics* myGraphics;

	vector<GameObject*> gameObjects;

	bool Frame();
	void InitializeWindows(int&, int&);
	void ShutdownWindows();
public:
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

#endif