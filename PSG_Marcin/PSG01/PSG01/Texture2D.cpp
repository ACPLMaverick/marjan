#include <Windows.h>
#include "Texture2D.h"
#include <fstream>

Texture2D::Texture2D(string fileName)
{
	this->fileName = fileName;
}


Texture2D::~Texture2D()
{
}

void Texture2D::bind()
{
	fstream f(fileName, ios::in | ios::binary);
	if (!f.is_open())
		die(1);
	BITMAPFILEHEADER bfh;
	f.read((char*)&bfh, sizeof(BITMAPFILEHEADER));
	if (f.gcount() != sizeof(BITMAPFILEHEADER))
		die(2);
	if (bfh.bfType != 'B' + ('M' << 8))
		die(3);
	BITMAPINFOHEADER bih;
	f.read((char*)&bih, sizeof(BITMAPINFOHEADER));
	if (f.gcount() != sizeof(BITMAPINFOHEADER))
		die(4);
	int width = bih.biWidth;
	int height = bih.biHeight;
	int dataSize = width*height*bih.biBitCount / 8;
	unsigned char *img;
	switch (bih.biBitCount)
	{
	case 32:
		img = new unsigned char[dataSize];
		f.read((char*)img, 16);
		if (f.gcount() != 16)
			die(5);
		break;
	case 24:
		img = new unsigned char[dataSize];
		break;
	default:
		die(6);
	}
	f.read((char*)img, dataSize);
	if (f.gcount() != dataSize)
		die(7);
}

void Texture2D::die(int)
{

}
