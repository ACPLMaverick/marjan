#include "Bitmap.h"


Bitmap::Bitmap()
{
	m_file = nullptr;
	m_ptr = nullptr;
	m_sourcePtr = nullptr;
	m_width = 0;
	m_height = 0;
	m_totalSize = 0;
}


Bitmap::~Bitmap()
{
	delete[] m_sourcePtr;
}

bool Bitmap::Load(const char* p)
{
	char type[2];
	unsigned int offset, w, h, totalSize;
	short bpp;

	m_file = fopen(p, "rb");

	if (m_file == nullptr)
		return false;

	// reading header
	fread(type, sizeof(char), 2, m_file);

	m_file->_ptr = m_file->_base;
	m_file->_ptr += 2;
	fread(&totalSize, sizeof(totalSize), 1, m_file);

	if (type[0] != 'B' || type[1] != 'M')
		return false;

	m_file->_ptr = m_file->_base;
	m_file->_ptr += 10;
	// reading pixel array offset
	fread(&offset, sizeof(offset), 1, m_file);

	// reading infoheader data
	m_file->_ptr = m_file->_base;
	m_file->_ptr += 18;
	fread(&w, sizeof(w), 1, m_file);

	m_file->_ptr = m_file->_base;
	m_file->_ptr += 22;
	fread(&h, sizeof(h), 1, m_file);

	m_file->_ptr = m_file->_base;
	m_file->_ptr += 28;
	fread(&bpp, sizeof(bpp), 1, m_file);

	if (bpp != 24)
	{
		return false;
	}

	m_width = w;
	m_height = h;
	m_dataSize = w * h * (bpp / 8);
	m_totalSize = totalSize;

	fclose(m_file);

	m_file = fopen(p, "rb");

	m_sourcePtr = new char[m_totalSize];

	fread(m_sourcePtr, sizeof(char), m_totalSize, m_file);

	m_ptr = (uchar3*)(m_sourcePtr + offset);

	fclose(m_file);

	return true;
}

bool Bitmap::Save(const char* p)
{
	FILE* destFile = fopen(p, "wb");
	if (destFile == nullptr)
		return false;

	fwrite(m_sourcePtr, sizeof(char), m_totalSize, destFile);

	fclose(destFile);

	return true;
}

uchar3* Bitmap::GetPtr()
{
	return m_ptr;
}

unsigned int Bitmap::GetWidth()
{
	return m_width;
}

unsigned int Bitmap::GetHeight()
{
	return m_height;
}
