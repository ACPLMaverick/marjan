#include "FPSCounter.h"


FPSCounter::FPSCounter()
{
}

FPSCounter::FPSCounter(const FPSCounter& other)
{
}

FPSCounter::~FPSCounter()
{
}

void FPSCounter::Initialize()
{
	m_fps = 0;
	m_count = 0;
	m_startTime = timeGetTime();
}

void FPSCounter::Frame()
{
	m_count++;

	if (timeGetTime() >= (m_startTime + 1000))
	{
		m_fps = m_count;
		m_count = 0;
		m_startTime = timeGetTime();
	}
}

int FPSCounter::GetFPS()
{
	return m_fps;
}