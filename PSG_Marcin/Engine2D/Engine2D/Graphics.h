#ifndef _GRAPHICS_H_
#define _GRAPHICS_H_

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
#include "ColorShader.h"

// globals
const bool FULL_SCREEN = false;
const bool SHOW_CURSOR = false;
const bool VSYNC_ENABLED = false;
const unsigned int BACKGROUND_COLOR = BLACK_BRUSH;
const float SCREEN_FAR = 1000.0f;
const float SCREEN_DEPTH = 0.1f;

class Graphics
{
private:
	Direct3D* m_D3D;
	Camera* m_Camera;
	vector<Model*> models;
	ColorShader* m_ColorShader;

	bool Render();
	bool InitializeModels();
	void RelaseModels();
public:
	Graphics();
	Graphics(const Graphics&);
	~Graphics();

	bool Initialize(int, int, HWND);
	void Shutdown();
	bool Frame();
};

#endif

