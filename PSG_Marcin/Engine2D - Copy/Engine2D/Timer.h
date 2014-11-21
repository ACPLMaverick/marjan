#pragma once

// includes
#include <Windows.h>

class Timer
{
private:
	INT64 m_frequency;
	float m_tickPerMs;
	INT64 m_startTime;
	float m_frameTime;
public:
	Timer();
	Timer(const Timer&);
	~Timer();

	bool Initialize();
	void Frame();

	float GetTime();
};

