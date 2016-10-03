#pragma once

/*
	System-wide defines
*/


//////////////////

// Other defines

//////////////////////////////////////
//////////////////////////////////////

class System;

class SystemSettings
{
	friend class System;

private:

#pragma region Constants

	const std::string C_LOADPATH = "Settings.ini";

#pragma endregion


#pragma region Settings

	std::string s_windowTitle = "FGK - Raytracer";
	int32_t s_windowWidth = 800;
	int32_t s_windowHeight = 600;
	bool s_windowVsync = false;
	bool s_windowFullscreen = false;
	bool s_soundEnabled = true;

	HINSTANCE _hInstance = NULL;
	LPWSTR _lpCmdLine = NULL;
	int _nCmdShow = 0;
	HWND _hwnd = NULL;

	int32_t _displayWidth = 0;
	int32_t _displayHeight = 0;

#pragma endregion

public:

	SystemSettings();
	~SystemSettings();

#pragma region Accessors

	std::string* GetWindowTitle() { return &s_windowTitle; }
	int32_t GetWindowWidth() { return s_windowWidth; }
	int32_t GetWindowHeight() { return s_windowHeight; }
	bool GetWindowVsync() { return s_windowVsync; }
	bool GetWindowFullscreen() { return s_windowFullscreen; }
	bool GetSoundEnabled() { return s_soundEnabled; }

	HINSTANCE GetHInstance() { return _hInstance; }
	LPWSTR* GetLpCmdLine() { return &_lpCmdLine; }
	int GetCmdShow() { return _nCmdShow; }
	HWND GetWindowPtr() { return _hwnd; }

	int32_t GetDisplayWidth() { return _displayWidth; }
	int32_t GetDisplayHeight() { return _displayHeight; }

#pragma endregion
};
