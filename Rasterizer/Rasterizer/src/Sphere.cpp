#include "Sphere.h"

Sphere::Sphere()
{
	this->_center = math::Float3();
	this->_radius = 1.0f;
}

Sphere::Sphere(math::Float3& c, float r)
{
	this->_center = c;
	this->_radius = r;
}

RayHit Sphere::CalcIntersect(Ray& ray)
{
	return RayHit();
}

void Sphere::Update()
{

}

void Sphere::Draw()
{

}