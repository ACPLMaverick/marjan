#pragma once
#include <glut.h>
#include <iostream>
#include <cstring>
#include <Windows.h>
#include "Texture2D.h"
using namespace std;

class Texture2D
{
private:
	string fileName;
	_Uint32t textureID;
public:
	Texture2D(string fileName);
	~Texture2D();

	void bind();
	void die(int);
};

