// Zad_1.cpp : Defines the entry point for the console application.
//

#include<cstdlib>
#include<Windows.h>

#include"Neuron.h"
#include"Network.h"

///GLOBALS///
HWND hwnd;
HDC theDC;

HANDLE semaphore;
Neuron n1, n2, n3;
Network network;

///METHODS///
DWORD WINAPI NeuronFunc(LPVOID)
{
	return 0;
}

void DrawWindow()
{
	theDC = GetDC(hwnd);
	TextOut(theDC, 45, 90, L"Sieæ Hopfielda", 30);
}

LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
	switch (msg)
	{
	case WM_KILLFOCUS:
		break;
	case WM_SETFOCUS:
		break;
	case WM_CLOSE:
		PostQuitMessage(0);
		break;
	case WM_PAINT:
		DrawWindow();
		break;
	case WM_DESTROY:
		CloseHandle(semaphore);
		for (int i = 0; i < 3; i++)
		{
			TerminateThread(network.neurons[i].my_thread, 0);
			CloseHandle(network.neurons[i].my_thread);
		}
		break;
	default:
		return DefWindowProc(hwnd, msg, wParam, lParam);
	}
	return 0;
}

void AppInit(HINSTANCE hInst, HINSTANCE hPrev, LPSTR szCmdLine, int sw)
{
	//Registering window class
	WNDCLASS cls;

	if (!hPrev)
	{
		cls.hCursor = LoadCursor(0, IDC_ARROW);
		cls.hIcon = NULL;
		cls.lpszMenuName = NULL;
		cls.lpszClassName = L"myWindowClass";
		cls.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
		cls.hInstance = hInst;
		cls.style = CS_VREDRAW | CS_HREDRAW;
		cls.lpfnWndProc = WndProc;
		cls.cbClsExtra = 0;
		cls.cbWndExtra = 0;
		if (!RegisterClass(&cls))
		{
			MessageBox(NULL, L"Window Registration Failed!", L"Error!",
				MB_ICONEXCLAMATION | MB_OK);
			return;
		}
	}

	//Creating application window
	RECT rek;
	GetWindowRect(GetDesktopWindow(), &rek);
	hwnd = CreateWindowEx(
		WS_EX_APPWINDOW,
		L"myWindowClass", L"Zadanie_1",
		WS_POPUP | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX | WS_MAXIMIZEBOX,
		((rek.right - rek.left) - 320) / 2, ((rek.bottom - rek.top) - 160) / 2,
		320, 160, 0, 0, hInst, 0);

	if (hwnd == NULL)
	{
		MessageBox(NULL, L"Window Creation Failed!", L"Error!",
			MB_ICONEXCLAMATION | MB_OK);
		return;
	}

	ShowWindow(hwnd, SW_SHOW);

	//Creating semaphore
	semaphore = CreateSemaphore(NULL, 0, 1, L"Przemek");

	//Creating threads
	DWORD ID;
	n1.my_thread = CreateThread(NULL, 0, NeuronFunc, 0, 0, &ID);
	n2.my_thread = CreateThread(NULL, 0, NeuronFunc, 0, 0, &ID);
	n3.my_thread = CreateThread(NULL, 0, NeuronFunc, 0, 0, &ID);
}

///MAIN///
int WinMain(HINSTANCE hInst, HINSTANCE hPrev, LPSTR szCmdLine, int sw)
{
	MSG msg;
	AppInit(hInst, hPrev, szCmdLine, sw);
	network = Network(&n1, &n2, &n3);
	for (;;)
	{
		if (PeekMessage(&msg, 0, 0, 0, PM_REMOVE))
		{
			if (msg.message == WM_QUIT)
			{
				break;
			}
			else
			{
				TranslateMessage(&msg);
				DispatchMessage(&msg);
			};
		}
		else
		{
			WaitMessage();
		}
	}
	//system("pause");
	return msg.wParam;
}

