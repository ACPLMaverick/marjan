#include "Triangle.h"



Triangle::Triangle(math::Float3 x, math::Float3 y, math::Float3 z, Color32 col) :
	Primitive()
{
	v1 = x;
	v2 = y;
	v3 = z;
	this->col = col;
}

RayHit Triangle::CalcIntersect(Ray& ray)
{
	return RayHit();
}


Triangle::~Triangle()
{
}

void Triangle::Update()
{
}