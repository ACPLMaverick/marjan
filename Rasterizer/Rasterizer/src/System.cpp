#define _CRT_SECURE_NO_WARNINGS
#include "stdafx.h"
#include "System.h"
#include "Scene.h"
#include "SceneTriangle.h"
#include "SceneMeshes.h"
#include "SceneSphere.h"
#include "SpecificObjectFactory.h"
#include "Timer.h"

// testing
#ifdef _DEBUG
#define _CRT_SECURE_NO_DEPRECATE
#include "Float3.h"
#include "Matrix4x4.h"
#include "Float4.h"
#endif // _DEBUG

#include <Commctrl.h>

#include <fcntl.h>
#include <stdio.h>
#include <io.h>
#include <string>

System::System()
{
}


System::~System()
{
}

void System::Initialize(HINSTANCE hInstance, LPWSTR lpCmdLine, int nCmdShow)
{
	_settings._hInstance = hInstance;
	_settings._lpCmdLine = lpCmdLine;
	_settings._nCmdShow = nCmdShow;

#ifdef _DEBUG

	std::cout.clear();

	AllocConsole();
	freopen("CONIN$", "r", stdin);
	freopen("CONOUT$", "w", stdout);
	freopen("CONOUT$", "w", stderr);

	std::cout.clear();
	std::ios::sync_with_stdio();
	std::cout << "Hi." << std::endl;
	/*
	// math tests
	math::Matrix4x4 m1(
		math::Float4(1.0f, 0.0f, 0.0f, 0.0f),
		math::Float4(2.0f, 1.0f, 0.0f, 0.0f),
		math::Float4(4.0f, 0.0f, 1.0f, 0.0f),
		math::Float4(6.0f, 0.0f, 0.0f, 1.0f)
		);

	math::Matrix4x4 m2(
		math::Float4(1.0f, 8.0f, 10.0f, 12.0f),
		math::Float4(0.0f, 1.0f, 0.0f, 0.0f),
		math::Float4(0.0f, 0.0f, 1.0f, 15.0f),
		math::Float4(0.0f, 0.0f, 0.0f, 1.0f)
	);

	math::Matrix4x4 m3 = m1 * m2;

	std::cout << m3;

	math::Float4 flt = math::Float4(1.0f, 3.0f, 5.0f, 7.0f);
	flt = m1 * flt;

	std::cout << flt;
	*/

	//math::Matrix4x4 lookAt, persp;
	//math::Matrix4x4::LookAt(&math::Float3(2.0f, 3.0f, 4.0f),
	//	&math::Float3(0.0f, 1.0f, 0.0f),
	//	&math::Float3(0.0f, 0.0f, -1.0f),
	//	&lookAt);

	//std::cout << lookAt << persp;
	/*
	//FGK_TESTY
	Sphere stefan = Sphere(math::Float3(0, 0, 0), 10);

	Ray robert1 = Ray(math::Float3(0.0f, 0.0f, -20.f), math::Float3(0.0f, 0.0f, 1.0f));
	Ray robert2 = Ray(math::Float3(0.0f, 0.0f, -20.f), math::Float3(0.0f, 1.0f, 0.0f));
	Ray robert3 = Ray(math::Float3(0.0f, -10.0f, -20.f), math::Float3(0.0f, 0.0f, 1.0f));
	RayHit hitRoberta1 = RayHit();
	RayHit hitRoberta2 = RayHit();
	RayHit hitRoberta3 = RayHit();

	hitRoberta1 = stefan.CalcIntersect(robert1);
	hitRoberta2 = stefan.CalcIntersect(robert2);
	hitRoberta3 = stefan.CalcIntersect(robert3);

	std::cout << "CZY PIERWSZY ROBERT TRAFIL? " << hitRoberta1.hit << " GDZIE? " << hitRoberta1.point.x << " " << hitRoberta1.point.y <<
		" " << hitRoberta1.point.z << std::endl;
	std::cout << "CZY DRUGI ROBERT TRAFIL?" << hitRoberta2.hit << std::endl;
	std::cout << "CZY TRZECI ROBERT TRAFIL? " << hitRoberta3.hit << " GDZIE? " << hitRoberta3.point.x << " " << hitRoberta3.point.y <<
		" " << hitRoberta3.point.z << std::endl;
	
	math::Float3 test = math::Float3(0.0f, (float)sqrt(2.0f), (float)sqrt(2.0f));
	math::Float3::Normalize(test);

	Plane przemek = Plane(math::Float3(0.0f, 0.0f, 0.0f), test);
	
	hitRoberta2 = przemek.CalcIntersect(robert2);

	std::cout << "CZY DRUGI ROBERT TRAFIL PRZEMKA?" << hitRoberta2.hit << " GDZIE? " << hitRoberta2.point.x << " " << 
		hitRoberta2.point.y << " " << hitRoberta2.point.z << std::endl;
	
	//KONIEC FGK TESTÓW
	*/

	//math::Matrix4x4 janusz;
	//math::Matrix4x4 waclaw;
	//math::Matrix4x4 zbychu, zdzichu;
	//math::Matrix4x4::LookAt(&math::Float3(0.0f, 0.0f, -5.0f),
	//		&math::Float3(0.0f, 0.0f, 0.0f),
	//		&math::Float3(0.0f, 1.0f, 0.0f),
	//		&janusz);
	//math::Matrix4x4::Perspective(
	//	45.0f,
	//	640.0f / 640.0f,
	//	0.01f,
	//	1000.0f,
	//	&waclaw
	//);
	//zbychu = waclaw * janusz;
	//zdzichu = janusz * waclaw;
	//std::cout << janusz << std::endl << waclaw << std::endl <<
	//	zbychu << std::endl << zdzichu << std::endl;
#endif // _DEBUG


	// initialize window in current OS
	InitWindow(hInstance, lpCmdLine, nCmdShow);

	// initialize console - debug only
	// initialize DIB
	RECT r;
	GetClientRect(_settings._hwnd, &r);
	
	ZeroMemory(&_bitmapScreenBufferInfo, sizeof(BITMAPINFO));
	_bitmapScreenBufferInfo.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
	_bitmapScreenBufferInfo.bmiHeader.biWidth = r.right - r.left;
	_bitmapScreenBufferInfo.bmiHeader.biHeight = r.bottom - r.top;
	_bitmapScreenBufferInfo.bmiHeader.biPlanes = 1;
	_bitmapScreenBufferInfo.bmiHeader.biBitCount = 32;
	_bitmapScreenBufferInfo.bmiHeader.biSizeImage = _settings.GetWindowWidth() * _settings.GetWindowHeight();
	_bitmapScreenBufferInfo.bmiHeader.biCompression = BI_RGB;
	_bitmapScreenBufferInfo.bmiHeader.biSizeImage = 0;

	HDC dc = GetWindowDC(_settings._hwnd);
	_bitmapScreenBuffer = CreateDIBSection(dc, &_bitmapScreenBufferInfo, DIB_RGB_COLORS, &_bitmapScreenBufferDataPtr, NULL, NULL);

	_settings._displayWidth = _bitmapScreenBufferInfo.bmiHeader.biWidth;
	_settings._displayHeight = _bitmapScreenBufferInfo.bmiHeader.biHeight;

	// initialize managers
	_renderer = SpecificObjectFactory::GetRenderer(&_settings);
	Timer::GetInstance()->Initialize();

	// initialize scenes
	//_scenes.push_back(new SceneTriangle());
	//std::string sName = "SceneTriangle";
	_scenes.push_back(new SceneMeshes());
	std::string sName = "SceneMeshes";
	//_scenes.push_back(new SceneSphere());
	//std::string sName = "SceneSpheres";
	_scenes[0]->Initialize(0, &sName);
}

void System::Shutdown()
{
#ifdef _DEBUG

	FreeConsole();

#endif // _DEBUG

	// shutdown managers
	delete _renderer;
	Timer::GetInstance()->Shutdown();
	Timer::GetInstance()->DestroyInstance();

	UnregisterClass((_settings.s_windowTitle.c_str()), _settings._hInstance);
	DestroyWindow(_settings._hwnd);
	PostQuitMessage(0);
}

void System::Run()
{
	while (_running)
	{
		RunMessages();

		// update timer
		Timer::GetInstance()->Update();

		// update scene instances
		_scenes[_currentScene]->Update();

		// draw scene to buffer
		_renderer->Draw(_scenes[_currentScene]);

		// fill bitmap with color buffer data
		size_t tj = _bitmapScreenBufferInfo.bmiHeader.biWidth;
		size_t ti = _bitmapScreenBufferInfo.bmiHeader.biHeight;
		for (size_t i = 0; i < ti; ++i)
		{
			for (size_t j = 0; j < tj; ++j)
			{
				((uint32_t*)_bitmapScreenBufferDataPtr)[i * tj + j] = (uint32_t)(_renderer->GetColorBuffer()->GetPixelScaled((uint16_t)j, (uint16_t)i, (uint16_t)tj, (uint16_t)ti).color);
			}
		}

		// draw bitmap in window
		RECT r;
		GetClientRect(_settings._hwnd, &r);
		InvalidateRect(_settings._hwnd, &r, false);
	}
}

void System::Pause()
{
}

void System::Stop()
{
	_running = false;
}

void System::AddEventHandlerMessage(std::function<void(UINT, WPARAM, LPARAM)>* func)
{
	_eventsMessage.push_back(func);
}

bool System::RemoveEventHandlerMessage(std::function<void(UINT, WPARAM, LPARAM)>* func)
{
	for (std::vector<std::function<void(UINT, WPARAM, LPARAM)>*>::iterator it = _eventsMessage.begin(); it != _eventsMessage.end(); ++it)
	{
		if (*it == func)
		{
			_eventsMessage.erase(it);
			return true;
		}
	}

	return false;
}

void System::InitWindow(HINSTANCE hInstance, LPWSTR lpCmdLine, int nCmdShow)
{
	WNDCLASS wc;
	wc.cbClsExtra = 0;
	wc.cbWndExtra = 0;
	wc.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);
	wc.hCursor = NULL;
	wc.hIcon = LoadIcon(NULL, IDI_WINLOGO);
	wc.hInstance = hInstance;
	wc.lpfnWndProc = WndProc;
	wc.lpszClassName = (_settings.s_windowTitle.c_str());
	wc.lpszMenuName = 0;
	wc.style = CS_HREDRAW | CS_VREDRAW;

	RegisterClass(&wc);

	_settings._hwnd = CreateWindow(
		(_settings.s_windowTitle.c_str()),
		(_settings.s_windowTitle.c_str()),
		(WS_OVERLAPPED |
			WS_CAPTION |
			WS_SYSMENU),
		10, 10,
		_settings.s_windowWidth, _settings.s_windowHeight,
		NULL, NULL,
		hInstance, NULL
		);

	// bottom bar for fps
	InitCommonControls();
	_settings._hwndStatus = CreateWindowEx(
		0,
		STATUSCLASSNAME,
		(PCTSTR)NULL,
		WS_CHILD | WS_VISIBLE,
		0, 0, 0, 0,
		_settings._hwnd,
		(HMENU)1,
		_settings._hInstance,
		NULL
	);

	ShowWindow(_settings._hwnd, nCmdShow);
	UpdateWindow(_settings._hwnd);
}

inline void System::RunMessages()
{
	MSG msg;

	while (PeekMessageW(&msg, _settings._hwnd, 0, 0, PM_REMOVE))
	{
		TranslateMessage(&msg);
		DispatchMessage(&msg);
	}
}

inline void System::ResizeWindowBitmap()
{
	if (_bitmapScreenBuffer == nullptr || _bitmapScreenBufferDataPtr == nullptr)
		return;

}

inline void System::DrawColorBuffer()
{
	if (_bitmapScreenBuffer == nullptr || _bitmapScreenBufferDataPtr == nullptr)
		return;

	PAINTSTRUCT ps;
	BITMAP nbm;
	HDC hdc = BeginPaint(_settings._hwnd, &ps);

	HDC hdcMem = CreateCompatibleDC(hdc);
	HBITMAP oldBitmap = (HBITMAP)SelectObject(hdcMem, (HBITMAP)_bitmapScreenBuffer);
	GetObject((HBITMAP)_bitmapScreenBuffer, sizeof(BITMAP), &nbm);
	BitBlt(hdc, 0, 0, ps.rcPaint.right - ps.rcPaint.left, ps.rcPaint.bottom - ps.rcPaint.top,
		hdcMem, 0, 0, SRCCOPY);

	SelectObject(hdcMem, oldBitmap);
	DeleteDC(hdcMem);

	EndPaint(_settings._hwnd, &ps);
}

inline void System::DrawFPS()
{
	std::string fps = std::to_string(Timer::GetInstance()->GetFPS());
	fps = fps.substr(0, 6);
	std::string fpsFormatted = "FPS: " + fps ;
	PAINTSTRUCT ps;
	HDC hdc = BeginPaint(_settings._hwndStatus, &ps);

	int result = DrawText(
		hdc,
		fpsFormatted.c_str(),
		-1,
		&ps.rcPaint,
		DT_LEFT | DT_BOTTOM
	);

	EndPaint(_settings._hwndStatus, &ps);
}

LRESULT System::WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
	// pass message to event handlers
	for (std::vector<std::function<void(UINT, WPARAM, LPARAM)>*>::iterator it = instance->_eventsMessage.begin(); it != instance->_eventsMessage.end(); ++it)
	{
		(*(*it))(message, wParam, lParam);
	}

	switch (message)
	{
	case WM_CREATE:
		break;
	case WM_SIZE:
		{
			System::GetInstance()->ResizeWindowBitmap();
		}

		break;
	case WM_PAINT:
		{
			System::GetInstance()->DrawColorBuffer();
			System::GetInstance()->DrawFPS();
		}
		
		break;
	case WM_DESTROY:
		System::GetInstance()->Stop();
		break;
	case WM_SETCURSOR:
		SetCursor(NULL);
		break;
	default:
		return DefWindowProc(hWnd, message, wParam, lParam);
		break;
	}

	return DefWindowProc(hWnd, message, wParam, lParam);
}