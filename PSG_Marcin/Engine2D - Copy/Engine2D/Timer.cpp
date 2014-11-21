#include "Timer.h"

Timer::Timer()
{
}

Timer::Timer(const Timer& other)
{
}

Timer::~Timer()
{
}

bool Timer::Initialize()
{
	// Check for support
	QueryPerformanceFrequency((LARGE_INTEGER*)&m_frequency);
	if (m_frequency == 0) return false;

	// how many ticks per milisecond
	m_tickPerMs = (float)(m_frequency / 1000);

	QueryPerformanceCounter((LARGE_INTEGER*)&m_startTime);
	return true;
}

void Timer::Frame()
{
	INT64 currentTime;
	float timeDifference;

	QueryPerformanceCounter((LARGE_INTEGER*)&currentTime);
	timeDifference = (float)(currentTime - m_startTime);
	m_frameTime = timeDifference / m_tickPerMs;
	m_startTime = currentTime;
}

float Timer::GetTime()
{
	return m_frameTime;
}