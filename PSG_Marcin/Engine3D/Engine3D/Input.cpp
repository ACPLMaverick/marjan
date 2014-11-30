#include "Input.h"


Input::Input()
{
	m_directInput = 0;
	m_keyboard = 0;
	m_mouse = 0;
}

Input::Input(const Input& other)
{
}

Input::~Input()
{
}

bool Input::Initialize(HINSTANCE hinstance, HWND hwnd, int width, int height)
{
	HRESULT result;
	m_screenHeight = height;
	m_screenWidth = width;
	m_mouseX = 0;
	m_mouseY = 0;

	result = DirectInput8Create(hinstance, DIRECTINPUT_VERSION, IID_IDirectInput8,
		(void**)&m_directInput, NULL);
	if (FAILED(result)) return false;

	result = m_directInput->CreateDevice(GUID_SysKeyboard, &m_keyboard, NULL);
	if (FAILED(result)) return false;

	result = m_keyboard->SetDataFormat(&c_dfDIKeyboard);
	if (FAILED(result)) return false;

	result = m_keyboard->SetCooperativeLevel(hwnd, DISCL_FOREGROUND | DISCL_EXCLUSIVE);
	if (FAILED(result)) return false;

	result = m_keyboard->Acquire();
	if (FAILED(result)) return false;

	result = m_directInput->CreateDevice(GUID_SysMouse, &m_mouse, NULL);
	if (FAILED(result)) return false;

	result = m_mouse->SetDataFormat(&c_dfDIMouse);
	if (FAILED(result)) return false;

	result = m_mouse->SetCooperativeLevel(hwnd, DISCL_FOREGROUND | DISCL_NONEXCLUSIVE);
	if (FAILED(result)) return false;

	result = m_mouse->Acquire();
	if (FAILED(result)) return false;

	return true;
}

void Input::Shutdown()
{
	// Release the mouse.
	if (m_mouse)
	{
		m_mouse->Unacquire();
		m_mouse->Release();
		m_mouse = 0;
	}

	// Release the keyboard.
	if (m_keyboard)
	{
		m_keyboard->Unacquire();
		m_keyboard->Release();
		m_keyboard = 0;
	}

	// Release the main interface to direct input.
	if (m_directInput)
	{
		m_directInput->Release();
		m_directInput = 0;
	}
}

bool Input::Frame()
{
	bool result;

	result = ReadKeyboard();
	if (!result) return false;

	result = ReadMouse();
	if (!result) return false;
	
	ProcessInput();

	return true;
}

bool Input::IsKeyDown(unsigned char key)
{
	if (m_keyboardState[key] & 0x80) return true;
	else return false;
}

void Input::GetMouseLocation(int& mouseX, int& mouseY)
{
	mouseX = m_mouseX;
	mouseY = m_mouseY;
}

bool Input::ReadKeyboard()
{
	HRESULT result;

	result = m_keyboard->GetDeviceState(sizeof(m_keyboardState), (LPVOID)&m_keyboardState);
	if (FAILED(result))
	{
		if ((result == DIERR_INPUTLOST) || (result == DIERR_NOTACQUIRED))
		{
			m_keyboard->Acquire();
		}
		else
		{
			return false;
		}
	}
	
	return true;
}

bool Input::ReadMouse()
{
	HRESULT result;

	result = m_mouse->GetDeviceState(sizeof(DIMOUSESTATE), (LPVOID)&m_mouseState);
	if (FAILED(result))
	{
		if ((result == DIERR_INPUTLOST) || (result == DIERR_NOTACQUIRED))
		{
			m_mouse->Acquire();
		}
		else
		{
			return false;
		}
	}

	return true;
}

void Input::ProcessInput()
{
	m_mouseX = m_mouseState.lX;
	m_mouseY = m_mouseState.lY;
}