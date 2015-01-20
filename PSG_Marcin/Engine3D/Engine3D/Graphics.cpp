#include "Graphics.h"


Graphics::Graphics()
{
	m_D3D = nullptr;
	m_Camera = nullptr;
	debugText = nullptr;
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

	result = m_D3D->Initialize(screenWidth, screenHeight, VSYNC_ENABLED, hwnd, FULL_SCREEN, SCREEN_DEPTH, SCREEN_NEAR);
	if (!result)
	{
		MessageBox(hwnd, "Could not initialize Direct3D", "Error", MB_OK);
		return false;
	}

	m_Camera = new Camera();
	if (!m_Camera) return false;
	m_Camera->SetPosition(D3DXVECTOR3(0.0f, 1.8f, -5.0f));
	m_Camera->SetTarget(D3DXVECTOR3(0.0f, 0.0f, 1.0f));
	m_Camera->Render();
	m_Camera->GetViewMatrix(baseViewMatrix);

	mFullScreenWindow = new OrthoWindow();
	mFullScreenWindow->Initialize(m_D3D->GetDevice(), SCREEN_WIDTH, SCREEN_HEIGHT);

	mDeferredBuffer = new DeferredBuffer();
	mDeferredBuffer->Initialize(m_D3D->GetDevice(), SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_DEPTH, SCREEN_NEAR);

	InitializeManagers(myHwnd);

	debugText = new Text();
	result = debugText->Initialize(m_D3D->GetDevice(), m_D3D->GetDeviceContext(), hwnd,
		screenWidth, screenHeight, baseViewMatrix);
	if (!result)
	{
		MessageBox(hwnd, "Could not initialize text object", "Error", MB_OK);
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
	if (debugText)
	{
		debugText->Shutdown();
		delete debugText;
		debugText = nullptr;
	}

	if (mDeferredBuffer)
	{
		mDeferredBuffer->Shutdown();
	}
	if (mFullScreenWindow)
	{
		mFullScreenWindow->Shutdown();
	}
}

bool Graphics::Frame(GameObject* objects[], unsigned int objectCount, Light* lights[])
{
	bool result;

	result = Render(objects, objectCount, lights);
	if (!result) return false;
	
	return true;
}

bool Graphics::Render(GameObject* objects[], unsigned int objectCount, Light* lights[])
{
	if (m_D3D && m_Camera)
	{
		D3DXMATRIX viewMatrix, projectionMatrix, worldMatrix, orthoMatrix;
		bool result;

		if (DEFERRED)
		{
			result = RenderSceneToTexture(objects, objectCount);
			if (!result) return false;

			m_D3D->BeginScene(0.0f, 0.0f, 0.0f, 1.0f);

			m_Camera->Render();

			m_Camera->GetViewMatrix(viewMatrix);
			m_D3D->GetWorldnMatrix(worldMatrix);
			m_D3D->GetOrthoMatrix(orthoMatrix);

			m_D3D->ZBufferOff();

			mFullScreenWindow->Render(m_D3D->GetDeviceContext());

			LightAmbient* ambient = (LightAmbient*)lights[0];
			int count = 0;
			for (; lights[count + 1] != nullptr && count <= LIGHT_MAX_COUNT; count++);
			D3DXVECTOR4 cols[LIGHT_MAX_COUNT];
			D3DXVECTOR4 dirs[LIGHT_MAX_COUNT];

			for (int i = 1; i <= count; i++)
			{
				cols[i - 1] = ((LightDirectional*)lights[i])->GetDiffuseColor();
				dirs[i - 1] = D3DXVECTOR4(((LightDirectional*)lights[i])->GetDirection().x, ((LightDirectional*)lights[i])->GetDirection().y, 
					((LightDirectional*)lights[i])->GetDirection().z, 0);
			}
			cols[0].w = count;

			D3DXVECTOR3 viewVector;
			D3DXVec3Subtract(&viewVector, &m_Camera->GetPosition(), &m_Camera->GetTarget());

			SpecularDeferredShader* shader = (SpecularDeferredShader*)shaderManager->LoadShader(m_D3D->GetDevice(), myHwnd, 5);
			result = shader->Render(m_D3D->GetDeviceContext(), mFullScreenWindow->GetIndexCount(), worldMatrix, baseViewMatrix, orthoMatrix, mDeferredBuffer->GetShaderResourceView(0),
				mDeferredBuffer->GetShaderResourceView(1), cols, dirs, count, ambient->GetDiffuseColor(), viewVector, D3DXVECTOR4(1.0f, 1.0f, 1.0f, 1.0f), 1.0f, 100.0f);
			
			m_D3D->ZBufferOn();

			m_D3D->EndScene();

			return true;
		}
		else
		{
			m_D3D->BeginScene(0.0f, 0.0f, 0.0f, 1.0f);


			m_Camera->GetViewMatrix(viewMatrix);
			m_Camera->Render();

			m_D3D->GetWorldnMatrix(worldMatrix);
			m_D3D->GetProjectionMatrix(projectionMatrix);
			m_D3D->GetOrthoMatrix(orthoMatrix);

			result = debugText->Render(m_D3D->GetDeviceContext(), worldMatrix, orthoMatrix);
			if (!result) return false;

			D3DXVECTOR3 viewVector;
			D3DXVec3Subtract(&viewVector, &m_Camera->GetPosition(), &m_Camera->GetTarget());

			GameObject* obj;
			for (int i = 0; i < objectCount; i++)
			{
				obj = objects[i];
				obj->Render(m_D3D->GetDeviceContext(), worldMatrix, viewMatrix, projectionMatrix, lights, viewVector);
			}
			delete[] objects;
			
			m_D3D->EndScene();

			return true;
		}
		
	}
	else return false;
}

bool Graphics::RenderSceneToTexture(GameObject* objects[], unsigned int objectCount)
{
	D3DXMATRIX worldMatrix, viewMatrix, projectionMatrix;

	mDeferredBuffer->SetRenderTargets(m_D3D->GetDeviceContext());
	mDeferredBuffer->ClearRenderTargets(m_D3D->GetDeviceContext(), 0.0f, 0.0f, 0.0f, 0.0f);

	m_Camera->GetViewMatrix(viewMatrix);

	m_D3D->GetWorldnMatrix(worldMatrix);
	m_D3D->GetProjectionMatrix(projectionMatrix);

	GameObject* obj;
	for (int i = 0; i < objectCount; i++)
	{
		obj = objects[i];
		obj->Render(m_D3D->GetDeviceContext(), worldMatrix, viewMatrix, projectionMatrix, NULL, D3DXVECTOR3(0.0f,0.0f,0.0f));
	}
	delete[] objects;

	m_D3D->SetBackBufferRenderTarget();
	m_D3D->ResetViewport();

	return true;
}

bool Graphics::InitializeManagers(HWND hwnd)
{
	textureManager = new TextureManager();

	textureManager->AddTexture(m_D3D->GetDevice(), "Assets/Textures/noTexture.dds");
	textureManager->AddTexture(m_D3D->GetDevice(), "Assets/Textures/test.dds");
	textureManager->AddTexture(m_D3D->GetDevice(), "Assets/Textures/moss_01_d.dds");
	textureManager->AddTexture(m_D3D->GetDevice(), "Assets/Textures/metal01_d.dds");
	textureManager->AddTexture(m_D3D->GetDevice(), "Assets/Textures/dynamiteCrate_diffuse.dds");
	textureManager->AddTexture(m_D3D->GetDevice(), "Assets/Textures/dynamiteCrate_spec.dds");

	shaderManager = new ShaderManager();
	bool result = shaderManager->AddShaders(m_D3D->GetDevice(), hwnd);
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
