#pragma once

#include "stdafx.h"
#include "Ray.h"
#include "Buffer.h"

class Primitive
{
protected:

#pragma region Protected


#pragma endregion

#pragma region Functions Protected


#pragma endregion

public:

#pragma region Functions Public

	Primitive();
	virtual ~Primitive();

	virtual RayHit CalcIntersect(Ray& ray) = 0;

	virtual void Update() = 0;
	virtual void Draw() = 0;

#pragma region Accessors

#pragma endregion

#pragma endregion
};

