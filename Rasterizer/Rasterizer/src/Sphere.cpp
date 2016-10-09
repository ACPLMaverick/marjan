#include "Sphere.h"

#include <stdio.h>

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
	math::Float3 ocVec = ray.GetOrigin() - _center;
	float B = -math::Float3::Dot(ray.GetDirection(), ocVec);
	float det = (B * B) - math::Float3::Dot(ocVec, ocVec) + (_radius * _radius);

	if (det > 0)
	{
		det = sqrt(det);
		float d1 = B + det;
		float d2 = B - det;

		if (d1 > 0)
		{
			if (d2 < 0)
			{
				//Ray origin inside sphere case
				return RayHit(true, math::Float3(ray.GetOrigin() + ray.GetDirection() * d1));
			}
			else
			{
				//Ray origin in front of sphere case
				return RayHit(true, math::Float3(ray.GetOrigin() + ray.GetDirection() * d2));
			}
		}
		else
		{
			return RayHit();
		}
	}
	else if (det == 0)
	{
		//Ray intersects only in one point (sphere tangent)
		return RayHit(true, math::Float3(ray.GetOrigin() + ray.GetDirection() * B));
	}
	else
	{
		//Ray doesn't intersect
		return RayHit();
	}
}

void Sphere::Update()
{

}

void Sphere::Draw()
{

}