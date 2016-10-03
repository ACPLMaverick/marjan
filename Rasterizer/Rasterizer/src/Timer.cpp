#include "stdafx.h"
#include "Timer.h"


Timer::Timer()
{
}


Timer::~Timer()
{
}

void Timer::Initialize()
{
	LARGE_INTEGER li;
	QueryPerformanceFrequency(&li);
	_freq = double(li.QuadPart);

	_startTime = GetCurrentTimeS();
	_currentTime = _startTime;
}

void Timer::Shutdown()
{
}

void Timer::Update()
{
	double time = GetCurrentTimeS();
	_deltaTime = time - _currentTime;
	_currentTime = time;

	_fps = 1.0f / max(_deltaTime, 0.00000000001);
}

double Timer::GetCurrentTimeS()
{
	double t;
	LARGE_INTEGER li;
	QueryPerformanceCounter(&li);
	t = double(li.QuadPart) / _freq - _startTime;
	return t;
}