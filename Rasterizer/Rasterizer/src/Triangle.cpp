#include "Triangle.h"



Triangle::Triangle(Float3 x, Float3 y, Float3 z, Color32 col) :
	Primitive()
{
	v1 = x;
	v2 = y;
	v3 = z;
	this->col = col;
}


Triangle::~Triangle()
{
}

void Triangle::Update()
{
}