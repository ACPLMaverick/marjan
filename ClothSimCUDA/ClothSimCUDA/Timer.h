#pragma once

/*
	This class's main purpose is calculate system total time, delta time per frame and number of frames per second.
*/

#include "Common.h"
#include "Singleton.h"

#include <GL\glew.h>
#include <GLFW\glfw3.h>
#include <chrono>

using namespace std::chrono;

class Timer : public Singleton<Timer>
{
	friend class Singleton<Timer>;

private:
	Timer();

	long double m_totalTime;
	long double m_deltaTime;
	long double m_fps;
public:
	Timer(const Timer*);
	~Timer();

	unsigned int Initialize();
	unsigned int Shutdown();
	unsigned int Run();

	double GetTotalTime();
	double GetDeltaTime();
	double GetFps();
};

