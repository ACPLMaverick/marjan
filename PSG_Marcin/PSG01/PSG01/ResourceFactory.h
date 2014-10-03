#pragma once
#include <map>
#include "Texture2D.h"
#include <iostream>
using namespace std;

class ResourceFactory
{
private:
	map<string, Texture2D> myMap;
public:
	ResourceFactory();
	~ResourceFactory();

	Texture2D* load(string path);
};

