#include "Graphics.h"


Graphics::Graphics()
{
	m_D3D = nullptr;
}

Graphics::Graphics(const Graphics& other)
{

}

Graphics::~Graphics()
{
}

bool Graphics::Initialize(int screenWidth, int screenHeight, HWND hwnd)
{
	bool result;

	m_D3D = new Direct3D();
	if (!m_D3D) return false;

	result = m_D3D->Initialize(screenWidth, screenHeight, VSYNC_ENABLED, hwnd, FULL_SCREEN, SCREEN_DEPTH, SCREEN_FAR);
	if (!result)
	{
		MessageBox(hwnd, "Could not initialize Direct3D", "Error", MB_OK);
		return false;
	}
	
	return true;
}

void Graphics::Shutdown()
{
	if (m_D3D)
	{
		m_D3D->Shutdown();
		delete m_D3D;
		m_D3D = nullptr;
	}
}

bool Graphics::Frame()
{
	bool result;

	result = Render();
	if (!result) return false;
	
	return true;
}

bool Graphics::Render()
{
	if (m_D3D)
	{
		m_D3D->BeginScene(0.5f, 0.5f, 0.5f, 1.0f);

		m_D3D->EndScene();

		return true;
	}
	else return false;
}
