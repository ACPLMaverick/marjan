#include "System.h"
#include "Bullet.h"

unsigned long System::frameCount;
bool System::playerAnimation;
float System::time;
unsigned long System::systemTime;

System::System()
{
	myInput = nullptr;
	myGraphics = nullptr;
	playerAnimation = false;
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
	myInput->Initialize(m_hinstance, m_hwnd, screenWidth, screenHeight);

	myGraphics = new Graphics();
	if (!myGraphics) return false;
	result = myGraphics->Initialize(screenWidth, screenHeight, m_hwnd);
	if (!result) return false;

	myScene = new Scene();
	myScene->Initialize(myGraphics, m_hwnd);

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
		myInput->Shutdown();
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

	if (myScene)
	{
		myScene->Shutdown();
		delete myScene;
		myScene = nullptr;
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

	if(!(myInput->Frame())) return false;

	if (!ProcessKeys()) return false;

	myScene->CheckGameObjects();
	ProcessCamera();

	/*vector<GameObject*> terVec = terrain->GetTiles();
	GameObject** goTab = new GameObject*[gameObjects.size() + terVec.size()];
	int i = 0;
	for (; i < terVec.size(); i++)
	{
		goTab[i] = terVec.at(i);
	}
	for (int j = 0; j < gameObjects.size(); i++, j++)
	{
		goTab[i] = gameObjects.at(j);
	}*/
	
	result = myGraphics->Frame(myScene->GetGameObjectsAsArray(), myScene->GetGameObjectsSize());
	if (!result) return false;

	return true;
}

bool System::ProcessKeys()
{
	bool toReturn = true;
	playerAnimation = false;
	Camera* cam = myGraphics->GetCamera();
	if (myInput->IsKeyDown(DIK_ESCAPE)) toReturn = false;
	if (myInput->IsKeyDown(DIK_A) /*&& player->GetPosition().x > -(signed int)(terrain->GetWidth()*terrain->GetSize())*/)
	{
		D3DXVECTOR3 target = cam->GetTarget();
		D3DXVECTOR3 newVec;
		D3DXVec3Normalize(&newVec, &target);
		RotateVector(newVec, D3DXVECTOR3(0.0f, 1.57079632679f, 0.0f));
		newVec.y = 0;
		D3DXVECTOR3 newerVec = cam->GetPosition() - (newVec*((myInput->movementDistance)*(m_Timer->GetTime())));
		
		cam->SetPosition(newerVec);
		playerAnimation = true;
		toReturn = true;
	}
	if (myInput->IsKeyDown(DIK_D))
	{
		D3DXVECTOR3 target = cam->GetTarget();
		D3DXVECTOR3 newVec;
		D3DXVec3Normalize(&newVec, &target);
		RotateVector(newVec, D3DXVECTOR3(0.0f, 1.57079632679f, 0.0f));
		newVec.y = 0;
		D3DXVECTOR3 newerVec = cam->GetPosition() + (newVec*((myInput->movementDistance)*(m_Timer->GetTime())));
		cam->SetPosition(newerVec);
		playerAnimation = true;
		toReturn = true;
	}
	if (myInput->IsKeyDown(DIK_W))
	{
		D3DXVECTOR3 target = cam->GetTarget();
		D3DXVECTOR3 newVec;
		D3DXVec3Normalize(&newVec, &target);
		D3DXVECTOR3 newerVec = cam->GetPosition() + (newVec*((myInput->movementDistance)*(m_Timer->GetTime())));
		cam->SetPosition(newerVec);
		playerAnimation = true;
		toReturn = true;
	}
	if (myInput->IsKeyDown(DIK_S))
	{
		D3DXVECTOR3 target = cam->GetTarget();
		D3DXVECTOR3 newVec;
		D3DXVec3Normalize(&newVec, &target);
		D3DXVECTOR3 newerVec = cam->GetPosition() - (newVec*((myInput->movementDistance)*(m_Timer->GetTime())));
		cam->SetPosition(newerVec);
		playerAnimation = true;
		toReturn = true;
	}
	
	if (myInput->IsKeyDown(DIK_SPACE))
	{
		/*PlayerShoot();
		myInput->KeyUp(VK_SPACE);
		toReturn = true;*/
	}
	return toReturn;
}

void System::ProcessCamera()
{
	Camera* cam = myGraphics->GetCamera();
	D3DXVECTOR3 lookAt;
	int mposx, mposy;
	float scale = 0.01f;
	myInput->GetMouseLocation(mposx, mposy);
	//lookAt = D3DXVECTOR3(cam->GetTarget().x + (float)mposx*scale, cam->GetTarget().y - (float)mposy*scale, cam->GetTarget().z);
	lookAt = cam->GetTarget();
	float fposx, fposy;
	fposx = mposx;
	fposy = mposy;

	// rotation
	D3DXMATRIX rotateX;
	D3DXMATRIX rotateY;
	D3DXMATRIX rotateZ;
	D3DXMatrixRotationX(&rotateX, mposy*scale*lookAt.z);
	D3DXMatrixRotationY(&rotateY, mposx*scale);
	D3DXMatrixRotationZ(&rotateZ, mposy*scale*(-lookAt.x));
	D3DXMATRIX rotationMatrix = rotateX*rotateY*rotateZ;
	D3DXVECTOR4 outputVec;

	D3DXVec3Transform(&outputVec, &lookAt, &rotationMatrix);
	lookAt.x = outputVec.x;
	lookAt.y = outputVec.y;
	lookAt.z = outputVec.z;

	cam->SetTarget(lookAt);
}

void System::PlayerShoot()
{
	GameObject* player = myScene->GetPlayer();
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
	myScene->Add(newBullet);
}

void System::RotateVector(D3DXVECTOR3& retVec, D3DXVECTOR3 rotationVector)
{
	// rotation
	D3DXMATRIX rotateX;
	D3DXMATRIX rotateY;
	D3DXMATRIX rotateZ;
	D3DXMatrixRotationX(&rotateX, rotationVector.x);
	D3DXMatrixRotationY(&rotateY, rotationVector.y);
	D3DXMatrixRotationZ(&rotateZ, rotationVector.z);
	D3DXMATRIX rotationMatrix = rotateX*rotateY*rotateZ;
	D3DXVECTOR4 outputVec;

	D3DXVec3Transform(&outputVec, &retVec, &rotationMatrix);
	retVec.x = outputVec.x;
	retVec.y = outputVec.y;
	retVec.z = outputVec.z;
}

LRESULT CALLBACK System::MessageHandler(HWND hwnd, UINT umsg, WPARAM wparam, LPARAM lparam)
{
	return DefWindowProc(hwnd, umsg, wparam, lparam);
}

void System::InitializeWindows(int& screenWidth, int& screenHeight)
{
	WNDCLASSEX wc;
	DEVMODE dmScreenSettings;
	int posX, posY;

	ApplicationHandle = this;
	m_hinstance = GetModuleHandle(NULL);
	applicationName = "Engine3D";

	//default settings
	wc.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;
	wc.lpfnWndProc = WndProc;
	wc.cbClsExtra = 0;
	wc.cbWndExtra = 0;
	wc.hInstance = m_hinstance;
	wc.hIcon = LoadIcon(NULL, IDI_SHIELD);
	wc.hIconSm = wc.hIcon;
	wc.hCursor = LoadCursor(NULL, IDC_NO);
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