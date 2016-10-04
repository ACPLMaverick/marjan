#include "stdafx.h"
#include "System.h"
#include "Scene.h"
#include "SceneTriangle.h"

#include "SpecificObjectFactory.h"

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

	// initialize window in current OS
	InitWindow(hInstance, lpCmdLine, nCmdShow);

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

	// initialize scenes
	_scenes.push_back(new SceneTriangle());
	std::string sName = "SceneTriangle";
	_scenes[0]->Initialize(0, &sName);
}

void System::Shutdown()
{
	// shutdown managers

	UnregisterClass((_settings.s_windowTitle.c_str()), _settings._hInstance);
	DestroyWindow(_settings._hwnd);
	PostQuitMessage(0);
}

void System::Run()
{
	while (_running)
	{
		RunMessages();

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
		WS_OVERLAPPEDWINDOW,
		10, 10,
		_settings.s_windowWidth, _settings.s_windowHeight,
		NULL, NULL,
		hInstance, NULL
		);

	int x = GetLastError();

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