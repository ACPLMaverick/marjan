// Tester.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#define L2_CACHE_SIZE 1024 * 1024
#define TEST_COUNT 10

struct Data16
{
	float x;
	float y;
	float z;
	float w;
};

struct Data20
{
	float x;
	float y;
	float z;
	float w;
	float a;
};

Data16 vectorData16[L2_CACHE_SIZE * 4];
Data16* listData16[L2_CACHE_SIZE * 4];
Data20 vectorData20[L2_CACHE_SIZE * 4];
Data20* listData20[L2_CACHE_SIZE * 4];

UINT32 Random32BitAddress()
{
	UINT32 x = rand() & 0xff;
	x |= (rand() & 0xff) << 8;
	x |= (rand() & 0xff) << 16;
	x |= (rand() & 0xff) << 24;
	return x;
}

double pcFreq = 0.0;
__int64 counterStart = 0;

void StartCounter()
{
	LARGE_INTEGER li;
	QueryPerformanceFrequency(&li);

	pcFreq = double(li.QuadPart) / 1000.0;

	QueryPerformanceCounter(&li);
	counterStart = li.QuadPart;
}

double GetCounter()
{
	LARGE_INTEGER li;
	QueryPerformanceCounter(&li);
	return double(li.QuadPart - counterStart) / pcFreq;
}

template<typename Data>
Data TestMemAccess(size_t sampleCount, Data* vector, Data** list, size_t size, double* seqTime, double* randTime)
{
	Data sample;

	// Generate random address list
	for (size_t i = 0; i < sampleCount; ++i)
	{
		size_t index = Random32BitAddress() % (sampleCount - 1);
		list[i] = &vector[index];
	}

	// Count access time for random address
	StartCounter();
	for (size_t i = 0; i < sampleCount; ++i)
	{
		memcpy(&sample, list[i], size);
	}
	*randTime += GetCounter();

	// Generate sequential address list
	for (size_t i = 0; i < sampleCount; ++i)
	{
		list[i] = &vector[i];
	}

	// Count access time for sequence address
	StartCounter();
	for (size_t i = 0; i < sampleCount; ++i)
	{
		memcpy(&sample, list[i], size);
	}
	*seqTime += GetCounter();

	return sample;
}

void Test(size_t sampleCount)
{
	srand(GetTickCount());
	int count = TEST_COUNT;
	double seqReadTimeData16 = 0.0;
	double randReadTimeData16 = 0.0;
	while (count--)
	{
		TestMemAccess<Data16>(sampleCount, vectorData16, listData16, sizeof(Data16), &seqReadTimeData16, &randReadTimeData16);
	}

	count = TEST_COUNT;
	double seqReadTimeData20 = 0.0;
	double randReadTimeData20 = 0.0;
	while (count--)
	{
		TestMemAccess<Data20>(sampleCount, vectorData20, listData20, sizeof(Data20), &seqReadTimeData20, &randReadTimeData20);
	}

	printf("TEST Data Packed 16\n");
	printf("Sequence read: %f ms\n", seqReadTimeData16/TEST_COUNT);
	printf("Random read: %f ms\n\n", randReadTimeData16 / TEST_COUNT);

	printf("TEST Data Packed 20\n");
	printf("Sequence read: %f ms\n", seqReadTimeData20 / TEST_COUNT);
	printf("Random read: %f ms\n\n", randReadTimeData20 / TEST_COUNT);
}

int _tmain(int argc, _TCHAR* argv[])
{
	printf("======= TEST 1/4 cache:\n");
	Test((L2_CACHE_SIZE / 4) / 16);
	printf("======= TEST 1/2 cache:\n");
	Test((L2_CACHE_SIZE / 2) / 16);
	printf("======= TEST full cache:\n");
	Test((L2_CACHE_SIZE ) / 16);
	printf("======= TEST 2 * cache:\n");
	Test((L2_CACHE_SIZE * 2) / 16);
	printf("======= TEST 4 * cache:\n");
	Test((L2_CACHE_SIZE * 4) / 16);

	getchar();

	return 0;
}

