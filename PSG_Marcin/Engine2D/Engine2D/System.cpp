#include "System.h"


System::System()
{
	myInput = nullptr;
	myGraphics = nullptr;
}


System::~System()
{
}

bool System::Initialize()
{
	int screenWidth = 0;
	int screenHeight = 0;
	bool result;

	InitializeWindows(screenWidth, screenHeight);

	myInput = new Input();
	if (!myInput) return false;
	myInput->Initialize();

	myGraphics = new Graphics();
	if (!myGraphics) return false;
	result = myGraphics->Initialize(screenWidth, screenHeight, m_hwnd);
	if (!result) return false;

	InitializeGameObjects();

	return true;
}

void System::Shutdown()
{
	if (myGraphics)
	{
		myGraphics->Shutdown();
		delete myGraphics;
		myGraphics = nullptr;
	}

	if (myInput)
	{
		delete myInput;
		myInput = nullptr;
	}

	for (std::vector<GameObject*>::iterator it = gameObjects.begin(); it != gameObjects.end(); ++it)
	{
		if (*it) (*it)->Destroy();
		delete (*it);
		(*it) = nullptr;
	}
	gameObjects.clear();

	ShutdownWindows();
}

void System::Run()
{
	MSG message;
	bool done, result;

	// initialize message structure
	ZeroMemory(&message, sizeof(MSG));

	// loop till theres a quit message from the window or user
	done = false;
	while (!done)
	{
		// handle windows messages
		if (PeekMessage(&message, NULL, 0, 0, PM_REMOVE))
		{
			TranslateMessage(&message);
			DispatchMessage(&message);
		}

		// exit when windows signals it
		if (message.message == WM_QUIT)
		{
			done = true;
		}
		else // if not then continue loop and do frame proc
		{
			result = Frame();
			if (!result) done = true;
		}
	}
}

// new frame processing funcionality will be placed here
bool System::Frame()
{
	bool result;

	if (myInput->IsKeyDown(VK_ESCAPE)) return false;

	GameObject** goTab = new GameObject*[gameObjects.size()];
	for (int i = 0; i < gameObjects.size(); i++)
	{
		goTab[i] = gameObjects.at(i);
	}
	result = myGraphics->Frame(goTab, gameObjects.size());
	if (!result) return false;
	return true;
}

void System::InitializeGameObjects()
{
	GameObject* go01 = new GameObject(
		"player", 
		"player", 
		(myGraphics->GetTextures())->LoadTexture(myGraphics->GetD3D()->GetDevice(), "./Assets/Textures/noTexture.dds"),
		(myGraphics->GetShaders())->LoadShader(myGraphics->GetD3D()->GetDevice(), m_hwnd, 0),
		myGraphics->GetD3D()->GetDevice(),
		D3DXVECTOR3(0.0f, 0.0f, 0.0f),
		D3DXVECTOR3(0.0f, 0.0f, 0.0f),
		D3DXVECTOR3(1.0f, 1.0f, 1.0f));
	gameObjects.push_back(go01);

	GameObject* go02 = new GameObject(
		"enemy",
		"enemies",
		(myGraphics->GetTextures())->LoadTexture(myGraphics->GetD3D()->GetDevice(), "./Assets/Textures/test.dds"),
		(myGraphics->GetShaders())->LoadShader(myGraphics->GetD3D()->GetDevice(), m_hwnd, 0),
		myGraphics->GetD3D()->GetDevice(),
		D3DXVECTOR3(-5.0f, 5.0f, 0.0f),
		D3DXVECTOR3(0.0f, 0.0f, 0.0f),
		D3DXVECTOR3(1.0f, 1.0f, 1.0f));
	gameObjects.push_back(go02);

	GameObject* go03 = new GameObject(
		"board",
		"map_nocollid",
		(myGraphics->GetTextures())->LoadTexture(myGraphics->GetD3D()->GetDevice(), "./Assets/Textures/moss_01_d.dds"),
		(myGraphics->GetShaders())->LoadShader(myGraphics->GetD3D()->GetDevice(), m_hwnd, 0),
		myGraphics->GetD3D()->GetDevice(),
		D3DXVECTOR3(0.0f, 0.0f, -1.0f),
		D3DXVECTOR3(0.0f, 0.0f, 0.0f),
		D3DXVECTOR3(15.0f, 10.0f, 1.0f));
	gameObjects.push_back(go03);
}

GameObject* System::GetGameObjectByName(LPCSTR name)
{
	for (std::vector<GameObject*>::iterator it = gameObjects.begin(); it != gameObjects.end(); ++it)
	{
		if ((*it)->GetName() == name) return (*it);
	}
	return nullptr;
}

void System::GetGameObjectsByTag(LPCSTR tag, GameObject** ptr, unsigned int &count)
{
	unsigned int c = 0;
	for (std::vector<GameObject*>::iterator it = gameObjects.begin(); it != gameObjects.end(); ++it)
	{
		if ((*it)->GetTag() == tag) c++;
	}
	ptr = new GameObject*[c];
	c = 0;
	for (std::vector<GameObject*>::iterator it = gameObjects.begin(); it != gameObjects.end(); ++it)
	{
		if ((*it)->GetTag() == tag)
		{
			ptr[c] = (*it);
			c++;
		}
	}
	count = c;
}

LRESULT CALLBACK System::MessageHandler(HWND hwnd, UINT umsg, WPARAM wparam, LPARAM lparam)
{
	switch (umsg)
	{
			case WM_KEYDOWN:
			{
				myInput->KeyDown((unsigned int)wparam);
				return 0;
			}
			case WM_KEYUP:
			{
				myInput->KeyUp((unsigned int)wparam);
				return 0;
			}
			default:
			{
				return DefWindowProc(hwnd, umsg, wparam, lparam);
			}
	}
}

void System::InitializeWindows(int& screenWidth, int& screenHeight)
{
	WNDCLASSEX wc;
	DEVMODE dmScreenSettings;
	int posX, posY;

	ApplicationHandle = this;
	m_hinstance = GetModuleHandle(NULL);
	applicationName = "Engine2D";

	//default settings
	wc.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;
	wc.lpfnWndProc = WndProc;
	wc.cbClsExtra = 0;
	wc.cbWndExtra = 0;
	wc.hInstance = m_hinstance;
	wc.hIcon = LoadIcon(NULL, IDI_WARNING);
	wc.hIconSm = wc.hIcon;
	wc.hCursor = LoadCursor(NULL, IDC_ARROW);
	wc.hbrBackground = (HBRUSH)GetStockObject(BACKGROUND_COLOR);
	wc.lpszMenuName = NULL;
	wc.lpszClassName = applicationName;
	wc.cbSize = sizeof(WNDCLASSEX);

	//register the window class
	RegisterClassEx(&wc);

	screenWidth = GetSystemMetrics(SM_CXSCREEN);
	screenHeight = GetSystemMetrics(SM_CYSCREEN);

	//screen settings
	if (FULL_SCREEN)
	{
		// set screen to max size of users desktop and 32bit
		memset(&dmScreenSettings, 0, sizeof(dmScreenSettings));
		dmScreenSettings.dmSize = sizeof(dmScreenSettings);
		dmScreenSettings.dmPelsWidth = (unsigned long)screenWidth;
		dmScreenSettings.dmPelsHeight = (unsigned long)screenHeight;
		dmScreenSettings.dmBitsPerPel = 32;
		dmScreenSettings.dmFields = DM_BITSPERPEL | DM_PELSWIDTH | DM_PELSHEIGHT;

		ChangeDisplaySettings(&dmScreenSettings, CDS_FULLSCREEN);
		
		// position of window in 0 - top left corner
		posX = posY = 0;
	}
	else
	{
		// 800x600 resolution
		screenWidth = 800;
		screenHeight = 600;

		// position window in the middle
		posX = (GetSystemMetrics(SM_CXSCREEN) - screenWidth) / 2;
		posY = (GetSystemMetrics(SM_CYSCREEN) - screenHeight) / 2;
	}

	// create window and get handle
	m_hwnd = CreateWindowEx(WS_EX_APPWINDOW, applicationName, applicationName,
		WS_CLIPSIBLINGS | WS_CLIPCHILDREN | WS_POPUP,
		posX, posY, screenWidth, screenHeight, NULL, NULL, m_hinstance, NULL);

	// bring up and set focus
	ShowWindow(m_hwnd, SW_SHOW);
	SetForegroundWindow(m_hwnd);
	SetFocus(m_hwnd);

	ShowCursor(SHOW_CURSOR);
}

void System::ShutdownWindows()
{
	ShowCursor(true);

	if (FULL_SCREEN) ChangeDisplaySettings(NULL, 0);
	DestroyWindow(m_hwnd);
	m_hwnd = NULL;

	// remove class instance
	UnregisterClass(applicationName, m_hinstance);
	m_hinstance = NULL;

	ApplicationHandle = NULL;
}

static LRESULT CALLBACK WndProc(HWND hwnd, UINT umessage, WPARAM wparam, LPARAM lparam)
{
	switch (umessage)
	{

		case WM_DESTROY:
		{
						   PostQuitMessage(0);
						   return 0;
		}

		case WM_CLOSE:
		{
						 PostQuitMessage(0);
						 return 0;
		}

		default:
		{
				   return ApplicationHandle->MessageHandler(hwnd, umessage, wparam, lparam);
		}
	}
}