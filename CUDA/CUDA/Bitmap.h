#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <fstream>

class Bitmap
{
private:
	FILE* m_file;
	uchar3* m_ptr;
	char* m_sourcePtr;
	unsigned int m_width;
	unsigned int m_height;
	unsigned int m_dataSize;
	unsigned int m_totalSize;
public:
	Bitmap();
	~Bitmap();

	bool Load(const char* p);
	bool Save(const char* p);

	uchar3* GetPtr();
	unsigned int GetWidth();
	unsigned int GetHeight();
};

