#include "Graphics.h"


Graphics::Graphics()
{
	m_D3D = nullptr;
	m_Camera = nullptr;
	m_TextureShader = nullptr;
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
	m_Camera->SetPosition(0.0f, 0.0f, -30.0f);

	result = InitializeModels();
	if (!result)
	{
		MessageBox(hwnd, "Could not initialize models!", "Error", MB_OK);
		return false;
	}

	m_TextureShader = new TextureShader();
	if (!m_TextureShader) return false;
	result = m_TextureShader->Initialize(m_D3D->GetDevice(), hwnd);
	if (!result)
	{
		MessageBox(hwnd, "Could not initialize the texture shader", "Error", MB_OK);
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
	RelaseModels();
	if (m_TextureShader)
	{
		m_TextureShader->Shutdown();
		delete m_TextureShader;
		m_TextureShader = nullptr;
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
	if (m_D3D && m_Camera && m_TextureShader)
	{
		D3DXMATRIX viewMatrix, projectionMatrix, worldMatrix;
		bool result;

		m_D3D->BeginScene(0.0f, 0.0f, 0.0f, 1.0f);

		m_Camera->Render();
		
		m_Camera->GetViewMatrix(viewMatrix);
		m_D3D->GetWorldnMatrix(worldMatrix);
		m_D3D->GetProjectionMatrix(projectionMatrix);

		for (std::vector<Model*>::iterator it = models.begin(); it != models.end(); ++it)
		{
			(*it)->Render(m_D3D->GetDeviceContext());
			result = m_TextureShader->Render(m_D3D->GetDeviceContext(), (*it)->GetIndexCount(), worldMatrix, viewMatrix, projectionMatrix, (*it)->GetTexture());
			if (!result) return false;
		}

		// put the model vertex and index buffers on the graphics pipeline to prepare them for drawing
		//m_Model->Render(m_D3D->GetDeviceContext());

		//// render the model using color shader
		//result = m_ColorShader->Render(m_D3D->GetDeviceContext(), m_Model->GetIndexCount(), worldMatrix, viewMatrix, projectionMatrix);
		//if (!result) return false;

		//m_Model02->Render(m_D3D->GetDeviceContext());
		//result = m_ColorShader->Render(m_D3D->GetDeviceContext(), m_Model02->GetIndexCount(), worldMatrix, viewMatrix, projectionMatrix);
		//if (!result) return false;

		m_D3D->EndScene();

		return true;
	}
	else return false;
}

bool Graphics::InitializeModels()
{
	// tu dodajemy nowe modele;
	Model* myModel;
	bool result;
	myModel = new Sprite2D(D3DXVECTOR3(0.0f, 0.0f, 0.0f), D3DXVECTOR3(0.0f, 0.0f, 0.0f), D3DXVECTOR3(1.0f, 1.0f, 1.0f), nullptr);
	if (!myModel) return false;
	result = myModel->Initialize(m_D3D->GetDevice(), "./Assets/Textures/noTexture.dds");
	if (!result) return false;
	models.push_back(myModel);

	myModel = new Sprite2D(D3DXVECTOR3(3.0f, 2.0f, 0.0f), D3DXVECTOR3(0.0f, 0.0f, 0.0f), D3DXVECTOR3(1.0f, 1.0f, 1.0f), nullptr);
	if (!myModel) return false;
	result = myModel->Initialize(m_D3D->GetDevice(), "./Assets/Textures/test.dds");
	if (!result) return false;
	models.push_back(myModel);

	myModel = new Sprite2D(D3DXVECTOR3(0.0f, -3.0f, 0.0f), D3DXVECTOR3(0.0f, 0.0f, 0.0f), D3DXVECTOR3(1.0f, 1.0f, 1.0f), nullptr);
	if (!myModel) return false;
	result = myModel->Initialize(m_D3D->GetDevice(), "./Assets/Textures/test.dds");
	if (!result) return false;
	models.push_back(myModel);

	return true;
}

void Graphics::RelaseModels()
{
	for (std::vector<Model*>::iterator it = models.begin(); it != models.end(); ++it)
	{
		if (*it) (*it)->Shutdown();
		delete (*it);
		(*it) = nullptr;
	}
	models.clear();
}
