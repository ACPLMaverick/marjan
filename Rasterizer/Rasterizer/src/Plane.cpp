#include "Plane.h"

Plane::Plane()
{
	this->_point = math::Float3();
	this->_normal = math::Float3(0, 0, 1.0f);
}

Plane::Plane(math::Float3& p, math::Float3& n)
{
	this->_point = p;
	this->_normal = n;
}

RayHit Plane::CalcIntersect(Ray& ray)
{
	return RayHit();
}

void Plane::Update()
{

}

void Plane::Draw()
{

}