#pragma once

#include "stdafx.h"
#include "Primitive.h"
#include "Float3.h"
#include "Color32.h"

/// <summary>
/// Winding direction - clockwise
/// </summary>
class Triangle :
	public Primitive
{
protected:

public:
	math::Float3 v1, v2, v3;
	Color32 col;

	Triangle(math::Float3 x, math::Float3 y, math::Float3 z, Color32 col = Color32());
	virtual ~Triangle();

	RayHit CalcIntersect(Ray& ray) override;
	virtual void Update() override;
	virtual void Draw() = 0;
};

