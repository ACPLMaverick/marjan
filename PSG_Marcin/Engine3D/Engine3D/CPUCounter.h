#pragma once
#pragma comment(lib, "pdh.lib")

// includes
#include <Pdh.h>

class CPUCounter
{
private:
	bool m_canReadCPU;
	HQUERY m_queryHandle;
	HCOUNTER m_counterHandle;
	unsigned long m_lastSampleTime;
	long m_cpuUsage;
public:
	CPUCounter();
	CPUCounter(const CPUCounter&);
	~CPUCounter();

	void Initialize();
	void Shutdown();
	void Frame();
	int GetCPUPercentage();
};

