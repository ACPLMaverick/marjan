#ifndef _GRAPHICSCLASS_H_
#define _GRAPHICSCLASS_H_

#include "d3dclass.h"
#include "cameraclass.h"
#include "bitmapclass.h"
#include "textureshaderclass.h"

#include <vector>

const bool FULL_SCREEN = false;
const bool VSYNC_ENABLED = true;
const float SCREEN_DEPTH = 1000.0F;
const float SCREEN_NEAR = 0.1f;

class GraphicsClass
{
public:
	GraphicsClass();

	bool Initialize(int, int, HWND);
	void Shutdown();
	bool Frame(int, int);
	BitmapClass* GetPlayer();

private:
	//bool Render(); gdy bez œwiat³a
	bool Render(int, int);
	bool InitializeTerrain(int, int, HWND, int, int);
	bool RenderTerrain(int, int, D3DXMATRIX, D3DXMATRIX, D3DXMATRIX);
	
	D3DClass* m_D3D;
	CameraClass* m_Camera;
	vector<BitmapClass*> m_Bitmaps, m_Terrain;
	TextureShaderClass* m_TextureShader;
};

#endif _GRAPHICSCLASS_H_