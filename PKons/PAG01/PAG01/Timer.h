#pragma once
#include <GLFW\glfw3.h>
#include <Windows.h>
using namespace std;

#define FLAG_ARRAY_SIZE 16
#define TIMER_RESOLUTION 1000

class Timer
{
private:
	double m_deltaTime;
	double m_currentTime;
	double m_timeStarted;
	double m_lastTime;
	double m_FPS;

	double flagArray[FLAG_ARRAY_SIZE];
public:
	Timer();
	~Timer();

	void Initialize();
	void Shutdown();
	void Update();

	double GetDeltaTime();
	double GetCurrentTime();
	double GetFPS();

	void SetFlag(unsigned int pos);
	double GetFlag(unsigned int pos);
};

