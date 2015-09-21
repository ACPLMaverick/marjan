#pragma once

/*
	This class's main purpose is calculate system total time, delta time per frame and number of frames per second.
*/

#include "Common.h"
#include "Singleton.h"

#include <GL\glew.h>
#include <GLFW\glfw3.h>
#include <map>
//#include <chrono>
//
//using namespace std::chrono;

class Timer : public Singleton<Timer>
{
	friend class Singleton<Timer>;

private:
	double m_totalTime;
	double m_deltaTime;
	double m_fps;

	std::map<unsigned int, double> m_timeStamps;

	Timer();
public:
	Timer(const Timer*);
	~Timer();

	unsigned int Initialize();
	unsigned int Shutdown();
	unsigned int Run();

	// If a stamp is in the collection, it is updated, otherwise, a new stamp is added to the collection.
	void AddTimeStamp(unsigned int);

	double GetTotalTime();
	double GetDeltaTime();
	double GetFps();
	
	double GetTimeStamp(unsigned int);
	// This function automatically removes stamp from collection.
	double GetTimeStampClear(unsigned int);
};

