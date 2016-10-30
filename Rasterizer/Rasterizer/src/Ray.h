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
	math::Float3 normal;
	math::Float2 uv;

	/// <summary>
	/// This is for teting
	/// </summary>
	int debugFlag = 0;

	RayHit() :
		hit(false),
		point(math::Float3()),
		barycentric(math::Float3()),
		normal(math::Float3(0.0f, 1.0f, 0.0f)),
		uv(math::Float2(0.5f, 0.5f))
	{
	}

	RayHit(bool isHit, math::Float3& p) :
		barycentric(math::Float3()),
		normal(math::Float3(0.0f, 1.0f, 0.0f)),
		uv(math::Float2())
	{
		this->hit = isHit;
		this->point = p;
	}

	RayHit(bool isHit, 
		math::Float3& p, 
		math::Float3& bar,
		math::Float3& nrm,
		math::Float2& uvv)
	{
		this->hit = isHit;
		this->point = p;
		this->barycentric = bar;
		this->normal = nrm;
		this->uv = uv;
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
	Ray(const math::Float3& s, const math::Float3& dir);

	math::Float3 GetOrigin();
	math::Float3 GetDirection();
};