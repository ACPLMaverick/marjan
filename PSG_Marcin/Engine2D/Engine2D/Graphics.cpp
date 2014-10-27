#include "Graphics.h"


Graphics::Graphics()
{
	m_D3D = nullptr;
	m_Camera = nullptr;
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
	myHwnd = hwnd;

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
	m_Camera->SetPosition(D3DXVECTOR3(0.0f, 0.0f, -30.0f));

	//TEMP_texture = new Texture();
	//TEMP_texture->Initialize(m_D3D->GetDevice(), "./Assets/Textures/noTexture.dds");

	InitializeManagers(myHwnd);

	//result = InitializeModels();
	//if (!result)
	//{
	//	MessageBox(hwnd, "Could not initialize models!", "Error", MB_OK);
	//	return false;
	//}
	
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
	//RelaseModels();
	if (textureManager)
	{
		textureManager->Shutdown();
		delete textureManager;
		textureManager = nullptr;
	}
	if (shaderManager)
	{
		shaderManager->Shutdown();
		delete shaderManager;
		shaderManager = nullptr;
	}
}

bool Graphics::Frame(GameObject* objects[], unsigned int objectCount)
{
	bool result;

	result = Render(objects, objectCount);
	if (!result) return false;
	
	return true;
}

bool Graphics::Render(GameObject* objects[], unsigned int objectCount)
{
	if (m_D3D && m_Camera)
	{
		D3DXMATRIX viewMatrix, projectionMatrix, worldMatrix;
		bool result;

		m_D3D->BeginScene(0.0f, 0.0f, 0.0f, 1.0f);

		m_Camera->Render();
		
		m_Camera->GetViewMatrix(viewMatrix);
		m_D3D->GetWorldnMatrix(worldMatrix);
		m_D3D->GetProjectionMatrix(projectionMatrix);

		GameObject* obj;
		for (int i = 0; i < objectCount; i++)
		{
			obj = objects[i];
			obj->Render(m_D3D->GetDeviceContext(), worldMatrix, viewMatrix, projectionMatrix);
		}
		delete[] objects;

		m_D3D->EndScene();

		return true;
	}
	else return false;
}

bool Graphics::InitializeManagers(HWND hwnd)
{
	textureManager = new TextureManager();

	textureManager->AddTexture(m_D3D->GetDevice(), "./Assets/Textures/noTexture.dds");
	textureManager->AddTexture(m_D3D->GetDevice(), "./Assets/Textures/test.dds");
	textureManager->AddTexture(m_D3D->GetDevice(), "./Assets/Textures/moss_01_d.dds");
	textureManager->AddTexture(m_D3D->GetDevice(), "./Assets/Textures/metal01_d.dds");
	textureManager->AddTexture(m_D3D->GetDevice(), "./Assets/Textures/bullet.dds");
	textureManager->AddTexture(m_D3D->GetDevice(), "./Assets/Textures/tank_player_FR_01.dds");
	textureManager->AddTexture(m_D3D->GetDevice(), "./Assets/Textures/tank_player_FR_02.dds");
	textureManager->AddTexture(m_D3D->GetDevice(), "./Assets/Textures/tank_player_FR_03.dds");
	textureManager->AddTexture(m_D3D->GetDevice(), "./Assets/Textures/tank_player_FR_04.dds");
	textureManager->AddTexture(m_D3D->GetDevice(), "./Assets/Textures/tank_player_FR_05.dds");
	textureManager->AddTexture(m_D3D->GetDevice(), "./Assets/Textures/tank_player_FR_06.dds");
	textureManager->AddTexture(m_D3D->GetDevice(), "./Assets/Textures/tank_player_FR_07.dds");
	textureManager->AddTexture(m_D3D->GetDevice(), "./Assets/Textures/tank_player_FR_08.dds");
	textureManager->AddTexture(m_D3D->GetDevice(), "./Assets/Textures/tank_player_FR_09.dds");
	textureManager->AddTexture(m_D3D->GetDevice(), "./Assets/Textures/tank_enemy_FR_01.dds");
	textureManager->AddTexture(m_D3D->GetDevice(), "./Assets/Textures/tank_enemy_FR_02.dds");
	textureManager->AddTexture(m_D3D->GetDevice(), "./Assets/Textures/tank_enemy_FR_03.dds");
	textureManager->AddTexture(m_D3D->GetDevice(), "./Assets/Textures/tank_enemy_FR_04.dds");
	textureManager->AddTexture(m_D3D->GetDevice(), "./Assets/Textures/tank_enemy_FR_05.dds");
	textureManager->AddTexture(m_D3D->GetDevice(), "./Assets/Textures/tank_enemy_FR_06.dds");
	textureManager->AddTexture(m_D3D->GetDevice(), "./Assets/Textures/tank_enemy_FR_07.dds");
	textureManager->AddTexture(m_D3D->GetDevice(), "./Assets/Textures/tank_enemy_FR_08.dds");
	textureManager->AddTexture(m_D3D->GetDevice(), "./Assets/Textures/tank_enemy_FR_09.dds");

	shaderManager = new ShaderManager();
	bool result = shaderManager->AddShader(m_D3D->GetDevice(), hwnd, "TextureShader");
	if (!result)
	{
		MessageBox(hwnd, "Could not initialize the texture shader", "Error", MB_OK);
		return false;
	}

	return true;
}

TextureManager* Graphics::GetTextures()
{
	return textureManager;
}

ShaderManager* Graphics::GetShaders()
{
	return shaderManager;
}

Direct3D* Graphics::GetD3D()
{
	return m_D3D;
}

Camera* Graphics::GetCamera()
{
	return m_Camera;
}
