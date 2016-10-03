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
	Float3 v1, v2, v3;
	Color32 col;

	Triangle(Float3 x, Float3 y, Float3 z, Color32 col = Color32());
	virtual ~Triangle();

	virtual void Update() override;
	virtual void Draw() = 0;
};

