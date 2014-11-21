#pragma once
#pragma comment(lib, "winmm.lib")

// includes
#include <windows.h>
#include <mmsystem.h>

class FPSCounter
{
private:
	int m_fps;
	int m_count;
	unsigned long m_startTime;
public:
	FPSCounter();
	FPSCounter(const FPSCounter&);
	~FPSCounter();

	void Initialize();
	void Frame();
	int GetFPS();
};

