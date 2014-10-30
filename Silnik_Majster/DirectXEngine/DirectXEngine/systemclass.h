#ifndef _SYSTEMCLASS_H_
#define _SYSTEMCLASS_H_

///////////////////////////////
// PRE-PROCESSING DIRECTIVES //
///////////////////////////////
#define WIN32_LEAN_AND_MEAN

//////////////
// INCLUDES //
//////////////
#include <windows.h>
#include <vector>

#include "d3dclass.h"
#include "inputclass.h"
#include "graphicsclass.h"

class SystemClass
{
public:
	SystemClass();

	bool Initialize();
	void Shutdown();
	void Run();

	/*to handle the windows system messages that will get sent to the application while it is running.*/
	LRESULT CALLBACK MessageHandler(HWND, UINT, WPARAM, LPARAM);

	int positionX = 256;
	int positionY = 256;
	float rotation = 0;
	WCHAR* currentTexture = L"../DirectXEngine/Data/Fruit0001_1_S.dds";

private:
	bool Frame();
	bool ProcessKeys();
	void InitializeWindows(int&, int&);
	void ShutdownWindows();

	LPCWSTR m_applicationName;
	HINSTANCE m_hinstance;
	HWND m_hwnd;

	/*pointers to the two objects that will handle graphics and input.*/
	InputClass* m_Input;
	GraphicsClass* m_Graphics;
	BitmapClass* m_Player;
};

static LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);
static SystemClass* ApplicationHandle = 0;

#endif SYSTEMCLASS_H