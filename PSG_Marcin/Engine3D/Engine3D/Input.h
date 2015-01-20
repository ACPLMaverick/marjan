#pragma once
#pragma comment(lib, "dinput8.lib")
#pragma comment(lib, "dxguid.lib")

#define DIRECTINPUT_VERSION 0x0800

#include <dinput.h>


class Input
{
private:
	IDirectInput8* m_directInput;
	IDirectInputDevice8* m_keyboard;
	IDirectInputDevice8* m_mouse;

	unsigned char m_keyboardState[256];
	DIMOUSESTATE m_mouseState;
	int m_screenWidth, m_screenHeight;
	float m_mouseX, m_mouseY;

	bool ReadKeyboard();
	bool ReadMouse();
	void ProcessInput();
public:
	float movementDistance = 0.01f;
	float prevMouseX, prevMouseY;

	Input();
	Input(const Input&);
	~Input();

	bool Initialize(HINSTANCE, HWND, int, int);
	void Shutdown();
	bool Frame();

	bool IsKeyDown(unsigned char);
	void GetMouseLocation(int&, int&);
};

