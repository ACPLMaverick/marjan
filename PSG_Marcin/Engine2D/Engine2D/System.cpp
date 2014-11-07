#include "System.h"
#include "Bullet.h"

unsigned long System::frameCount;
bool System::playerAnimation;
unsigned int System::checkGameObjects;
float System::time;
unsigned long System::systemTime;

System::System()
{
	myInput = nullptr;
	myGraphics = nullptr;
	playerAnimation = false;
	checkGameObjects = false;
	m_CPU = nullptr;
	m_FPS = nullptr;
	m_Timer = nullptr;
}


System::~System()
{
}

bool System::Initialize()
{
	int screenWidth = 0;
	int screenHeight = 0;
	bool result;
	frameCount = 0;

	InitializeWindows(screenWidth, screenHeight);

	myInput = new Input();
	if (!myInput) return false;
	myInput->Initialize();

	myGraphics = new Graphics();
	if (!myGraphics) return false;
	result = myGraphics->Initialize(screenWidth, screenHeight, m_hwnd);
	if (!result) return false;

	InitializeGameObjects();
	InitializeTerrain();

	m_FPS = new FPSCounter();
	m_FPS->Initialize();

	m_CPU = new CPUCounter();
	m_CPU->Initialize();

	m_Timer = new Timer();
	m_Timer->Initialize();
	time = m_Timer->GetTime();

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

	if (m_CPU)
	{
		m_CPU->Shutdown();
		delete m_CPU;
		m_CPU = nullptr;
	}

	if (m_FPS)
	{
		delete m_FPS;
		m_FPS = nullptr;
	}

	if (m_Timer)
	{
		delete m_Timer;
		m_Timer = nullptr;
	}

	for (std::vector<GameObject*>::iterator it = gameObjects.begin(); it != gameObjects.end(); ++it)
	{
		if (*it) (*it)->Destroy();
		delete (*it);
		(*it) = nullptr;
	}
	gameObjects.clear();

	if (terrain)
	{
		terrain->Shutdown();
		delete terrain;
		terrain = nullptr;
	}

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
			frameCount++;
			if (!result) done = true;
		}
	}
}

// new frame processing funcionality will be placed here
bool System::Frame()
{
	bool result;

	systemTime = timeGetTime();
	m_Timer->Frame();
	m_FPS->Frame();
	m_CPU->Frame();
	frameCount++;

	time = m_Timer->GetTime();

	if (!ProcessKeys()) return false;

	CheckGameObjects();
	ProcessCamera();

	vector<GameObject*> terVec = terrain->GetTiles();
	GameObject** goTab = new GameObject*[gameObjects.size() + terVec.size()];
	int i = 0;
	for (; i < terVec.size(); i++)
	{
		goTab[i] = terVec.at(i);
	}
	for (int j = 0; j < gameObjects.size(); i++, j++)
	{
		goTab[i] = gameObjects.at(j);
	}
	result = myGraphics->Frame(goTab, gameObjects.size() + terVec.size());
	if (!result) return false;

	return true;
}

bool System::ProcessKeys()
{
	//string debug = to_string(player->position.x) + " " + to_string(player->position.y) + "\n";
	//OutputDebugString(debug.c_str());
	bool toReturn = true;
	playerAnimation = false;
	if (myInput->IsKeyDown(VK_ESCAPE)) toReturn = false;
	if (myInput->IsKeyDown(VK_LEFT) /*&& player->GetPosition().x > -(signed int)(terrain->GetWidth()*terrain->GetSize())*/)
	{
		D3DXVECTOR3 newVec = (player->GetPosition() + D3DXVECTOR3((-myInput->movementDistance)*(m_Timer->GetTime()), 0.0f, 0.0f));
		player->SetPosition(newVec);
		player->SetRotation(D3DXVECTOR3(0.0f, 0.0f, 1.57079632679f));
		//myInput->KeyUp(VK_LEFT);
		playerAnimation = true;
		toReturn = true;
	}
	if (myInput->IsKeyDown(VK_RIGHT))
	{
		D3DXVECTOR3 newVec = (player->GetPosition() + D3DXVECTOR3((myInput->movementDistance)*(m_Timer->GetTime()), 0.0f, 0.0f));
		player->SetPosition(newVec);
		player->SetRotation(D3DXVECTOR3(0.0f, 0.0f, 4.71238898038f));
		//myInput->KeyUp(VK_RIGHT);
		playerAnimation = true;
		toReturn = true;
	}
	if (myInput->IsKeyDown(VK_UP))
	{
		D3DXVECTOR3 newVec = (player->GetPosition() + D3DXVECTOR3(0.0f, (myInput->movementDistance)*(m_Timer->GetTime()), 0.0f));
		player->SetPosition(newVec);
		player->SetRotation(D3DXVECTOR3(0.0f, 0.0f, 0.0f));
		//myInput->KeyUp(VK_UP);
		playerAnimation = true;
		toReturn = true;
	}
	if (myInput->IsKeyDown(VK_DOWN))
	{
		D3DXVECTOR3 newVec = (player->GetPosition() + D3DXVECTOR3(0.0f, (-myInput->movementDistance)*(m_Timer->GetTime()), 0.0f));
		player->SetPosition(newVec);
		player->SetRotation(D3DXVECTOR3(0.0f, 0.0f, 3.14159265359f));
		//myInput->KeyUp(VK_DOWN);
		playerAnimation = true;
		toReturn = true;
	}
	if (myInput->IsKeyDown(VK_DOWN) && myInput->IsKeyDown(VK_LEFT))
	{
		player->SetRotation(D3DXVECTOR3(0.0f, 0.0f, 2.35619449019f));
	}
	if (myInput->IsKeyDown(VK_DOWN) && myInput->IsKeyDown(VK_RIGHT))
	{
		player->SetRotation(D3DXVECTOR3(0.0f, 0.0f, 3.92699081699f));
	}
	if (myInput->IsKeyDown(VK_UP) && myInput->IsKeyDown(VK_LEFT))
	{
		player->SetRotation(D3DXVECTOR3(0.0f, 0.0f, 0.78539816339f));
	}
	if (myInput->IsKeyDown(VK_UP) && myInput->IsKeyDown(VK_RIGHT))
	{
		player->SetRotation(D3DXVECTOR3(0.0f, 0.0f, 5.49778714378f));
	}
	if (myInput->IsKeyDown(VK_SPACE))
	{
		PlayerShoot();
		myInput->KeyUp(VK_SPACE);
		toReturn = true;
	}
	return toReturn;
}

void System::ProcessCamera()
{
	Camera* cam = myGraphics->GetCamera();
	D3DXVECTOR3 newPos;
	newPos.x = player->GetPosition().x;
	newPos.y = player->GetPosition().y;
	newPos.z = cam->GetPosition().z;
	D3DXVECTOR3 targetPos = player->GetPosition();
	targetPos.z += 1.0f;
	
	cam->SetPosition(newPos);
	//cam->SetTarget(targetPos);
}

void System::InitializeGameObjects()
{
	Texture* g01Tex[9];
	g01Tex[0] = (myGraphics->GetTextures())->LoadTexture(myGraphics->GetD3D()->GetDevice(), "./Assets/Textures/tank_player_FR_01.dds");
	g01Tex[1] = (myGraphics->GetTextures())->LoadTexture(myGraphics->GetD3D()->GetDevice(), "./Assets/Textures/tank_player_FR_02.dds");
	g01Tex[2] = (myGraphics->GetTextures())->LoadTexture(myGraphics->GetD3D()->GetDevice(), "./Assets/Textures/tank_player_FR_03.dds");
	g01Tex[3] = (myGraphics->GetTextures())->LoadTexture(myGraphics->GetD3D()->GetDevice(), "./Assets/Textures/tank_player_FR_04.dds");
	g01Tex[4] = (myGraphics->GetTextures())->LoadTexture(myGraphics->GetD3D()->GetDevice(), "./Assets/Textures/tank_player_FR_05.dds");
	g01Tex[5] = (myGraphics->GetTextures())->LoadTexture(myGraphics->GetD3D()->GetDevice(), "./Assets/Textures/tank_player_FR_06.dds");
	g01Tex[6] = (myGraphics->GetTextures())->LoadTexture(myGraphics->GetD3D()->GetDevice(), "./Assets/Textures/tank_player_FR_07.dds");
	g01Tex[7] = (myGraphics->GetTextures())->LoadTexture(myGraphics->GetD3D()->GetDevice(), "./Assets/Textures/tank_player_FR_08.dds");
	g01Tex[8] = (myGraphics->GetTextures())->LoadTexture(myGraphics->GetD3D()->GetDevice(), "./Assets/Textures/tank_player_FR_09.dds");
	GameObject* go01 = new GameObject(
		"player", 
		"player", 
		g01Tex,
		9,
		(myGraphics->GetShaders())->LoadShader(myGraphics->GetD3D()->GetDevice(), m_hwnd, 0),
		myGraphics->GetD3D()->GetDevice(),
		D3DXVECTOR3(0.0f, 0.0f, 0.0f),
		D3DXVECTOR3(0.0f, 0.0f, 0.0f),
		D3DXVECTOR3(1.0f, 1.0f, 1.0f));
	player = go01;
	gameObjects.push_back(go01);

	Texture* enemyTex[9];
	enemyTex[0] = (myGraphics->GetTextures())->LoadTexture(myGraphics->GetD3D()->GetDevice(), "./Assets/Textures/tank_enemy_FR_01.dds");
	enemyTex[1] = (myGraphics->GetTextures())->LoadTexture(myGraphics->GetD3D()->GetDevice(), "./Assets/Textures/tank_enemy_FR_02.dds");
	enemyTex[2] = (myGraphics->GetTextures())->LoadTexture(myGraphics->GetD3D()->GetDevice(), "./Assets/Textures/tank_enemy_FR_03.dds");
	enemyTex[3] = (myGraphics->GetTextures())->LoadTexture(myGraphics->GetD3D()->GetDevice(), "./Assets/Textures/tank_enemy_FR_04.dds");
	enemyTex[4] = (myGraphics->GetTextures())->LoadTexture(myGraphics->GetD3D()->GetDevice(), "./Assets/Textures/tank_enemy_FR_05.dds");
	enemyTex[5] = (myGraphics->GetTextures())->LoadTexture(myGraphics->GetD3D()->GetDevice(), "./Assets/Textures/tank_enemy_FR_06.dds");
	enemyTex[6] = (myGraphics->GetTextures())->LoadTexture(myGraphics->GetD3D()->GetDevice(), "./Assets/Textures/tank_enemy_FR_07.dds");
	enemyTex[7] = (myGraphics->GetTextures())->LoadTexture(myGraphics->GetD3D()->GetDevice(), "./Assets/Textures/tank_enemy_FR_08.dds");
	enemyTex[8] = (myGraphics->GetTextures())->LoadTexture(myGraphics->GetD3D()->GetDevice(), "./Assets/Textures/tank_enemy_FR_09.dds");
	GameObject* go02 = new GameObject(
		"enemy01",
		"enemies",
		enemyTex,
		9,
		(myGraphics->GetShaders())->LoadShader(myGraphics->GetD3D()->GetDevice(), m_hwnd, 0),
		myGraphics->GetD3D()->GetDevice(),
		D3DXVECTOR3(-5.0f, 5.0f, 0.0f),
		D3DXVECTOR3(0.0f, 0.0f, 3.14159265359f),
		D3DXVECTOR3(1.0f, 1.0f, 1.0f));
	gameObjects.push_back(go02);
	GameObject* go02b = new GameObject(
		"enemy02",
		"enemies",
		enemyTex,
		9,
		(myGraphics->GetShaders())->LoadShader(myGraphics->GetD3D()->GetDevice(), m_hwnd, 0),
		myGraphics->GetD3D()->GetDevice(),
		D3DXVECTOR3(0.0f, 5.0f, 0.0f),
		D3DXVECTOR3(0.0f, 0.0f, 3.14159265359f),
		D3DXVECTOR3(1.0f, 1.0f, 1.0f));
	gameObjects.push_back(go02b);
	GameObject* go02c = new GameObject(
		"enemy03",
		"enemies",
		enemyTex,
		9,
		(myGraphics->GetShaders())->LoadShader(myGraphics->GetD3D()->GetDevice(), m_hwnd, 0),
		myGraphics->GetD3D()->GetDevice(),
		D3DXVECTOR3(3.0f, 5.0f, 0.0f),
		D3DXVECTOR3(0.0f, 0.0f, 3.14159265359f),
		D3DXVECTOR3(1.0f, 1.0f, 1.0f));
	gameObjects.push_back(go02c);
	GameObject* go12 = new GameObject(
		"test_bullet",
		"bullets",
		(myGraphics->GetTextures())->LoadTexture(myGraphics->GetD3D()->GetDevice(), "./Assets/Textures/bullet.dds"),
		(myGraphics->GetShaders())->LoadShader(myGraphics->GetD3D()->GetDevice(), m_hwnd, 0),
		myGraphics->GetD3D()->GetDevice(),
		D3DXVECTOR3(-4.5f, 5.5f, 0.1f),
		D3DXVECTOR3(0.0f, 0.0f, 0.0f),
		D3DXVECTOR3(1.0f, 1.0f, 1.0f));
	gameObjects.push_back(go12);
	go12->SetTransparency(0.6f);
	//player = go12;
}

void System::InitializeTerrain()
{
	Texture* terrainTextures[3];
	terrainTextures[0] = (myGraphics->GetTextures())->LoadTexture(myGraphics->GetD3D()->GetDevice(), "./Assets/Textures/metal01_d.dds");
	terrainTextures[1] = (myGraphics->GetTextures())->LoadTexture(myGraphics->GetD3D()->GetDevice(), "./Assets/Textures/moss_01_d.dds");
	terrainTextures[2] = (myGraphics->GetTextures())->LoadTexture(myGraphics->GetD3D()->GetDevice(), "./Assets/Textures/test.dds");
	terrain = new Terrain("Configs/TerrainProperties.xml", myGraphics->GetTextures(), (myGraphics->GetShaders())->LoadShader(myGraphics->GetD3D()->GetDevice(), m_hwnd, 0), myGraphics->GetD3D());
	terrain->Initialize();
}

void System::CheckGameObjects()
{
	while (checkGameObjects > 0)
	{
		for (std::vector<GameObject*>::iterator it = gameObjects.begin(); it != gameObjects.end(); ++it)
		{
			if ((*it)->GetDestroySignal())
			{
				(*it)->Destroy();
				delete (*it);
				(*it) = nullptr;
				gameObjects.erase(it);
				break;
			}
		}
		checkGameObjects--;
	}
}

void System::PlayerShoot()
{
	Bullet* newBullet = new Bullet(
		"test_bullet",
		"bullets",
		(myGraphics->GetTextures())->LoadTexture(myGraphics->GetD3D()->GetDevice(), "./Assets/Textures/bullet.dds"),
		(myGraphics->GetShaders())->LoadShader(myGraphics->GetD3D()->GetDevice(), m_hwnd, 0),
		myGraphics->GetD3D()->GetDevice(),
		(player->GetPosition() + D3DXVECTOR3(0.0f,0.0f,0.1f)),
		player->GetRotation(),
		D3DXVECTOR3(0.3f, 0.3f, 0.3f),
		5.0f,
		50.0f);
	gameObjects.push_back(newBullet);
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
	wc.hIcon = LoadIcon(NULL, IDI_SHIELD);
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