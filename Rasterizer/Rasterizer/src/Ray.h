#pragma once

#include "stdafx.h"

struct RayHit
{
public:
	bool hit;
	math::Float3 point;

	RayHit()
	{
		hit = false;
		point = math::Float3();
	}

	RayHit(bool isHit, math::Float3& p)
	{
		this->hit = isHit;
		this->point = p;
	}
};

class Ray
{
protected:
	math::Float3 _origin;
	math::Float3 _dir;

	float _distance;

public:
	Ray();
	Ray(math::Float3& s, math::Float3& dir);

	math::Float3 GetOrigin();
	math::Float3 GetDirection();
};