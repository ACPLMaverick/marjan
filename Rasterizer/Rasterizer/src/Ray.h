#pragma once

#include "stdafx.h"

struct RayHit
{
	bool hit;
	math::Float3 point;

	/// <summary>
	/// This is only valid for ray-triangle intersection.
	/// </summary>
	math::Float3 barycentric;

	/// <summary>
	/// This is for teting
	/// </summary>
	int debugFlag = 0;

	RayHit() :
		hit(false),
		point(math::Float3()),
		barycentric(math::Float3())
	{
	}

	RayHit(bool isHit, math::Float3& p) :
		barycentric(math::Float3())
	{
		this->hit = isHit;
		this->point = p;
	}

	RayHit(bool isHit, math::Float3& p, math::Float3& bar)
	{
		this->hit = isHit;
		this->point = p;
		this->barycentric = bar;
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