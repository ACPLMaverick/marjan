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

	//Create the bitmap object
	m_Bitmaps.push_back(new BitmapClass);
	if (!m_Bitmaps.at(0))
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
	int tilesCountHeight = (int)(screenHeight / bitmapHeight);
	int tilesCountWidth = (int)(screenWidth / bitmapWidth);
	int tilesCount = tilesCountHeight*tilesCountWidth;
	for (int i = 0; i < tilesCount; i++)
	{
		m_Terrain.push_back(new BitmapClass);
		if (!m_Bitmaps.at(i))
		{
			return false;
		}

		result = m_Terrain.at(i)->Initialize(m_D3D->GetDevice(), screenWidth, screenHeight, L"../DirectXEngine/Data/Grass0129_9_S.dds", bitmapWidth, bitmapHeight);
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

bool GraphicsClass::Frame(int positionX, int positionY)
{
	bool result;
	static float rotation = 0.0f;

	//Update the rotation variable each frame
	rotation += (float)D3DX_PI*0.01f;
	if (rotation > 360.0f)
	{
		rotation -= 360.0f;
	}

	//Render the graphics scene
	result = Render(positionX, positionY);
	if (!result)
	{
		return false;
	}

	return true;
}

bool GraphicsClass::Render(int positionX, int positionY)
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

	//Rotate the world matrix by the rotation value so thtat the triangle will spin
	//D3DXMatrixRotationY(&worldMatrix, rotation);

	//Turn off the Z buffer to begin all 2D rendering
	m_D3D->TurnZBufferOff();

	//Put the bitmap vertex and index buffers on the graphics pipeline to prepare them for drawing
	result = m_Bitmaps.at(0)->Render(m_D3D->GetDeviceContext(), positionX, positionY);
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

bool GraphicsClass::RenderTerrain(int bitmapWidth, int bitmapHeight, D3DXMATRIX worldMatrix, D3DXMATRIX viewMatrix, D3DXMATRIX orthoMatrix)
{
	return true;
}

BitmapClass* GraphicsClass::GetPlayer()
{
	return m_Bitmaps.at(0);
}