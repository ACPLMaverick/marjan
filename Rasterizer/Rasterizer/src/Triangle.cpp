#include "Triangle.h"



Triangle::Triangle(math::Float3 x, math::Float3 y, math::Float3 z, 
	math::Float3 ux, math::Float3 uy, math::Float3 uz,
	math::Float3 cx, math::Float3 cy, math::Float3 cz,
	Color32 col) :
	Primitive()
{
	v1 = x;
	v2 = y;
	v3 = z;
	u1 = ux;
	u2 = uy;
	u3 = uz;
	c1 = cx;
	c2 = cy;
	c3 = cz;
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