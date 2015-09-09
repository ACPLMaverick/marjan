#include "Timer.h"

Timer::Timer()
{
	m_deltaTime = 0.0;
	m_fps = 0.0;
	m_totalTime = 0.0;
}

Timer::Timer(const Timer*)
{
}

Timer::~Timer()
{
}



unsigned int Timer::Initialize()
{
	m_start = high_resolution_clock::now();
	return CS_ERR_NONE;
}

unsigned int Timer::Shutdown()
{
	return CS_ERR_NONE;
}

unsigned int Timer::Run()
{
	high_resolution_clock::time_point point = high_resolution_clock::now();
	duration<long double> timeSpan = duration_cast<duration<long double>>(point - m_start);
	
	long double newTime = timeSpan.count() * 1000.0;
	m_deltaTime = newTime - m_totalTime;
	m_totalTime = newTime;

	m_fps = 1.0 / (m_deltaTime / 1000.0);

	//printf("%f, %f, %f\n", m_totalTime, m_deltaTime, m_fps);

	return CS_ERR_NONE;
}



double Timer::GetTotalTime()
{
	return m_totalTime;
}

double Timer::GetDeltaTime()
{
	return m_deltaTime;
}

double Timer::GetFps()
{
	return m_fps;
}