#include "Timer.h"


Timer::Timer()
{
	m_deltaTime = 0;
	m_currentTime = 0;
	m_lastTime = 0;
	m_FPS = 0;
	m_timeStarted = 0;
	for (int i = 0; i < FLAG_ARRAY_SIZE; i++)
		flagArray[i] = 0;
}


Timer::~Timer()
{
}

void Timer::Initialize()
{
	m_timeStarted = glfwGetTime() * TIMER_RESOLUTION;
	m_currentTime = glfwGetTime() * TIMER_RESOLUTION;
	m_lastTime = glfwGetTime() * TIMER_RESOLUTION;
}

void Timer::Shutdown()
{

}

void Timer::Update()
{
	m_lastTime = m_currentTime;
	m_currentTime = glfwGetTime() * TIMER_RESOLUTION;
	m_deltaTime = m_currentTime - m_lastTime;
	m_FPS = TIMER_RESOLUTION / m_deltaTime;
}

double Timer::GetDeltaTime()
{
	return m_deltaTime;
}

double Timer::GetCurrentTime()
{
	return m_currentTime;
}

double Timer::GetFPS()
{
	return m_FPS;
}

void Timer::SetFlag(unsigned int pos)
{
	if (pos >= FLAG_ARRAY_SIZE)
	{
		return;
	}
	flagArray[pos] = glfwGetTime() * TIMER_RESOLUTION;
}

double Timer::GetFlag(unsigned int pos)
{
	if (pos >= FLAG_ARRAY_SIZE)
	{
		return 0;
	}
	return flagArray[pos];
}
