#include "CPUCounter.h"


CPUCounter::CPUCounter()
{
}

CPUCounter::CPUCounter(const CPUCounter& other)
{
}

CPUCounter::~CPUCounter()
{
}

void CPUCounter::Initialize()
{
	PDH_STATUS status;

	m_canReadCPU = true;

	status = PdhOpenQuery(NULL, 0, &m_queryHandle);
	if (status != ERROR_SUCCESS) m_canReadCPU = false;

	// set query object to poll all cpus in the system
	status = PdhAddCounter(m_queryHandle, TEXT("\\Processor(_Total)\\% processor time"), 0, &m_counterHandle);
	if (status != ERROR_SUCCESS) m_canReadCPU = false;

	m_lastSampleTime = GetTickCount();
	m_cpuUsage = 0;
}

void CPUCounter::Shutdown()
{
	if (m_canReadCPU) PdhCloseQuery(m_queryHandle);
}

void CPUCounter::Frame()
{
	PDH_FMT_COUNTERVALUE value;

	if (m_canReadCPU)
	{
		if ((m_lastSampleTime + 1000) < GetTickCount())
		{
			m_lastSampleTime = GetTickCount();
			PdhCollectQueryData(m_queryHandle);
			PdhGetFormattedCounterValue(m_counterHandle, PDH_FMT_LONG, NULL, &value);
			m_cpuUsage = value.longValue;
		}
	}
}

int CPUCounter::GetCPUPercentage()
{
	int usage;

	if (m_canReadCPU) usage = (int)m_cpuUsage;
	else usage = 0;
	return usage;
}