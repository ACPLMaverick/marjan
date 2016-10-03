#pragma once

#include "Primitive.h"

class Plane : public Primitive
{
protected:
	math::Float3 _point;
	math::Float3 _normal;

public:	
	Plane();
	Plane(math::Float3& p, math::Float3& n);

	RayHit CalcIntersect(Ray& ray) override;

	void Update() override;
	void Draw() override;

};