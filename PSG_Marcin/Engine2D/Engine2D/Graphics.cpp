#include "Graphics.h"


Graphics::Graphics()
{
	m_D3D = nullptr;
	m_Camera = nullptr;
	m_Model = nullptr;
	m_Model02 = nullptr;
	m_ColorShader = nullptr;
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

	m_Camera = new Camera();
	if (!m_Camera) return false;
	m_Camera->SetPosition(0.0f, 0.0f, -10.0f);

	m_Model = new Model(D3DXVECTOR3(3.0f, 0.0f, 0.0f));
	if (!m_Model) return false;
	result = m_Model->Initialize(m_D3D->GetDevice());
	if (!result)
	{
		MessageBox(hwnd, "Could not initialize the model", "Error", MB_OK);
		return false;
	}

	m_Model02 = new Model();
	if (!m_Model02) return false;
	result = m_Model02->Initialize(m_D3D->GetDevice());
	if (!result)
	{
		MessageBox(hwnd, "Could not initialize the model 02", "Error", MB_OK);
		return false;
	}

	m_ColorShader = new ColorShader();
	if (!m_ColorShader) return false;
	result = m_ColorShader->Initialize(m_D3D->GetDevice(), hwnd);
	if (!result)
	{
		MessageBox(hwnd, "Could not initialize the color shader", "Error", MB_OK);
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
	if (m_Camera)
	{
		delete m_Camera;
		m_Camera = nullptr;
	}
	if (m_Model)
	{
		m_Model->Shutdown();
		delete m_Model;
		m_Model = nullptr;
	}
	if (m_Model02)
	{
		m_Model02->Shutdown();
		delete m_Model02;
		m_Model02 = nullptr;
	}
	if (m_ColorShader)
	{
		m_ColorShader->Shutdown();
		delete m_ColorShader;
		m_ColorShader = nullptr;
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
	if (m_D3D && m_Model && m_Camera && m_ColorShader)
	{
		D3DXMATRIX viewMatrix, projectionMatrix, worldMatrix;
		bool result;

		m_D3D->BeginScene(0.5f, 0.5f, 0.5f, 1.0f);

		m_Camera->Render();
		
		m_Camera->GetViewMatrix(viewMatrix);
		m_D3D->GetWorldnMatrix(worldMatrix);
		m_D3D->GetProjectionMatrix(projectionMatrix);

		// put the model vertex and index buffers on the graphics pipeline to prepare them for drawing
		m_Model->Render(m_D3D->GetDeviceContext());

		// render the model using color shader
		result = m_ColorShader->Render(m_D3D->GetDeviceContext(), m_Model->GetIndexCount(), worldMatrix, viewMatrix, projectionMatrix);
		if (!result) return false;

		m_Model02->Render(m_D3D->GetDeviceContext());
		result = m_ColorShader->Render(m_D3D->GetDeviceContext(), m_Model->GetIndexCount(), worldMatrix, viewMatrix, projectionMatrix);
		if (!result) return false;

		m_D3D->EndScene();

		return true;
	}
	else return false;
}
