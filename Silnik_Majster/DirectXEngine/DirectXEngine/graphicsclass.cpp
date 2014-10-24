#include "graphicsclass.h"

GraphicsClass::GraphicsClass() /*initialize the pointer to null for safety reasons*/
{
	m_D3D = 0;
	m_Camera = 0;
	m_TextureShader = 0;
}

bool GraphicsClass::Initialize(int screenWidth, int screenHeight, HWND hwnd)
{
	bool result;

	//Create the Direct3D object
	m_D3D = new D3DClass;
	if (!m_D3D)
	{
		return false;
	}

	//Initialize the Direct3D object
	result = m_D3D->Initialize(screenWidth, screenHeight, VSYNC_ENABLED, hwnd, FULL_SCREEN, SCREEN_DEPTH, SCREEN_NEAR);
	if (!result)
	{
		MessageBox(hwnd, L"Could not initialize Direct3D", L"Error", MB_OK);
		return false;
	}

	//Create the camera object
	m_Camera = new CameraClass;
	if (!m_Camera)
	{
		return false;
	}

	//Set the initial position of the camera
	m_Camera->SetPosition(0.0f, 0.0f, -10.f);

	LoadTextures();

	//Create the bitmap object
	m_Bitmaps.push_back(new BitmapClass);
	if (!m_Bitmaps.at(0))
	{
		return false;
	}

	if (!InitializeTerrain(screenWidth, screenHeight, hwnd, 256, 256))
	{
		return false;
	}

	//Initialize the bitmap object
	result = m_Bitmaps.at(0)->Initialize(m_D3D->GetDevice(), screenWidth, screenHeight, L"../DirectXEngine/Data/Grass0129_9_S.dds", 256, 256);
	if (!result)
	{
		MessageBox(hwnd, L"Could not initialize the bitmap object", L"Error", MB_OK);
		return false;
	}

	// Create the texture shader object.
	m_TextureShader = new TextureShaderClass;
	if (!m_TextureShader)
	{
		return false;
	}

	// Initialize the texture shader object.
	result = m_TextureShader->Initialize(m_D3D->GetDevice(), hwnd);
	if (!result)
	{
		MessageBox(hwnd, L"Could not initialize the texture shader object.", L"Error", MB_OK);
		return false;
	}

	return true;
}

bool GraphicsClass::InitializeTerrain(int screenWidth, int screenHeight, HWND hwnd, int bitmapWidth, int bitmapHeight)
{
	bool result;
	int tilesCount = 12;
	for (int i = 0; i < tilesCount; i++)
	{
		m_Terrain.push_back(new BitmapClass);
		if (!m_Terrain.at(i))
		{
			return false;
		}

		result = m_Terrain.at(i)->Initialize(m_D3D->GetDevice(), screenWidth, screenHeight, L"../DirectXEngine/Data/FloorsCheckerboard0017_9_S.dds", bitmapWidth, bitmapHeight);
		if (!result)
		{
			return false;
		}
	}

	return true;
}

void GraphicsClass::Shutdown() /* Shut down of all graphics objects occur here */
{
	if (m_Bitmaps.size() >= 1)
	{
		for (int i = 0; i < m_Bitmaps.size(); i++)
		{
			m_Bitmaps.at(i)->Shutdown();
			delete m_Bitmaps.at(i);
			m_Bitmaps.at(i) = 0;
		}
	}

	if (m_Terrain.size() >= 1)
	{
		for (int i = 0; i < m_Terrain.size(); i++)
		{
			m_Terrain.at(i)->Shutdown();
			delete m_Terrain.at(i);
			m_Terrain.at(i) = 0;
		}
	}

	if (m_TextureShader)
	{
		m_TextureShader->Shutdown();
		delete m_TextureShader;
		m_TextureShader = 0;
	}

	DeleteTextures(); 

	if (m_Camera)
	{
		delete m_Camera;
		m_Camera = 0;
	}

	if (m_D3D)
	{
		m_D3D->Shutdown();
		delete m_D3D;
		m_D3D = 0;
	}

	return;
}

bool GraphicsClass::Frame(int positionX, int positionY, float rotation, WCHAR* currentTexture)
{
	bool result;

	m_Bitmaps.at(0)->LoadTexture(m_D3D->GetDevice(), currentTexture);
	if (positionY > 0)
	{
		m_Camera->SetPosition(positionX - 256, -positionY + 256, -10.0f);
	}
	else
	m_Camera->SetPosition(positionX - 256, abs(positionY - 256), -10.0f);

	//Render the graphics scene
	result = Render(positionX, positionY, rotation);
	if (!result)
	{
		return false;
	}

	return true;
}

bool GraphicsClass::Render(int positionX, int positionY, float rotation)
{
	D3DXMATRIX viewMatrix, worldMatrix, projectionMatrix, orthoMatrix;
	bool result;

	//Clear the buffers to begin the scene
	m_D3D->BeginScene(0.0f, 0.0f, 0.0f, 1.0f);

	//Generate the view matrix based on the camera's position
	m_Camera->Render();

	//Get the world, view and projection matrices from the camera and d3d objects
	m_Camera->GetViewMatrix(viewMatrix);
	m_D3D->GetWorldMatrix(worldMatrix);
	m_D3D->GetProjectionMatrix(projectionMatrix);
	m_D3D->GetOrthoMatrix(orthoMatrix);

	//Rotate the world matrix by the rotation value so that the triangle will spin
	//D3DXMatrixRotationZ(&worldMatrix, rotation);

	//Turn off the Z buffer to begin all 2D rendering
	m_D3D->TurnZBufferOff();

	if (!RenderTerrain(worldMatrix, viewMatrix, orthoMatrix))
	{
		return false;
	}

	//Put the bitmap vertex and index buffers on the graphics pipeline to prepare them for drawing
	result = m_Bitmaps.at(0)->Render(m_D3D->GetDeviceContext(), positionX, positionY, worldMatrix, rotation);
	if (!result)
	{
		return false;
	}

	// Render the model using the texture shader.
	result = m_TextureShader->Render(m_D3D->GetDeviceContext(), m_Bitmaps.at(0)->GetIndexCount(), worldMatrix, viewMatrix, orthoMatrix,
		m_Bitmaps.at(0)->GetTexture());
	if (!result)
	{
		return false;
	}

	//Turn the Z buffer back on now that all 2D rendering has completed
	m_D3D->TurnZBufferOn();

	//Present the endered scene to the screen
	m_D3D->EndScene();

	return true;
}

bool GraphicsClass::RenderTerrain(D3DXMATRIX worldMatrix, D3DXMATRIX viewMatrix, D3DXMATRIX orthoMatrix)
{
	bool result;
	int terrainPositionX = 0;
	int terrainPositionY = 0;
	float rotation = 0;

	for (int i = 0; i < m_Terrain.size(); i++)
	{
		result = m_Terrain.at(i)->Render(m_D3D->GetDeviceContext(), terrainPositionX, terrainPositionY, worldMatrix, rotation);
		if (!result)
		{
			return false;
		}

		// Render the model using the texture shader.
		result = m_TextureShader->Render(m_D3D->GetDeviceContext(), m_Terrain.at(i)->GetIndexCount(), worldMatrix, viewMatrix, orthoMatrix,
			m_Terrain.at(i)->GetTexture());
		if (!result)
		{
			return false;
		}

		terrainPositionX += 256;
		if (terrainPositionX >= 1024)
		{
			terrainPositionY += 256;
			terrainPositionX = 0;
		}
	}

	return true;
}

BitmapClass* GraphicsClass::GetPlayer()
{
	return m_Bitmaps.at(0);
}

void GraphicsClass::LoadTextures()	//Dodaæ wczytywanie z xmla
{
	textures.push_back(L"../DirectXEngine/Data/Fruit0001_1_S.dds");
	textures.push_back(L"../DirectXEngine/Data/Grass0129_9_S.dds");

	return;
}

void GraphicsClass::DeleteTextures()
{
	textures.pop_back();
	textures.pop_back();

	return;
}