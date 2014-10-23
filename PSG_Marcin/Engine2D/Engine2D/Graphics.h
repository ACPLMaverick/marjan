#pragma once
//defines
#define DKGRAY_BRUSH 3
#define GRAY_BRUSH 2
//includes
#include <Windows.h>
#include <vector>
// my classes
#include "Direct3D.h"
#include "Camera.h"
#include "Model.h"
#include "Sprite2D.h"
#include "TextureShader.h"
#include "TextureManager.h"
#include "ShaderManager.h"
#include "GameObject.h"

// globals
const bool FULL_SCREEN = false;
const bool SHOW_CURSOR = false;
const bool VSYNC_ENABLED = true;
const unsigned int BACKGROUND_COLOR = BLACK_BRUSH;
const float SCREEN_FAR = 1000.0f;
const float SCREEN_DEPTH = 0.1f;

class GameObject;

class Graphics
{
private:
	Direct3D* m_D3D;
	Camera* m_Camera;
	vector<Model*> models;
	HWND myHwnd;

	TextureManager* textureManager;
	ShaderManager* shaderManager;

	bool Render(GameObject* objects[], unsigned int objectCount);
	//bool InitializeModels();
	bool InitializeManagers(HWND hwnd);
	//void RelaseModels();
public:
	Graphics();
	Graphics(const Graphics&);
	~Graphics();

	bool Initialize(int, int, HWND);
	void Shutdown();
	bool Frame(GameObject* objects[], unsigned int objectCount);

	TextureManager* GetTextures();
	ShaderManager* GetShaders();
	Direct3D* GetD3D();
	Camera* GetCamera();
};
