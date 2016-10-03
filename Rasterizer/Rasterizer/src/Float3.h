#pragma once

#include "stdafx.h"

namespace math
{
	struct Float3
	{
		//union
		//{
		//	float tab[3];
		float x, y, z;
		//};

		Float3()
		{
			x = 0.0f;
			y = 0.0f;
			z = 0.0f;
		}

		Float3(const Float3& c)
		{
			this->x = c.x;
			this->y = c.y;
			this->z = c.z;
		}

		Float3(float x, float y, float z)
		{
			this->x = x;
			this->y = y;
			this->z = z;
		}

		Float3 operator*(const Float3& right)
		{
			return Float3(this->x * right.x, this->y * right.y, this->z * right.z);
		}

		Float3 operator*(float scalar)
		{
			return Float3(this->x * scalar, this->y * scalar, this->z * scalar);
		}

		Float3 operator/(float scalar)
		{
			Float3 f = Float3();
			float inverse = 1.0f / scalar;

			f.x *= inverse;
			f.y *= inverse;
			f.z *= inverse;

			return f;
		}

		Float3 operator+(const Float3& right)
		{
			return Float3(this->x + right.x, this->y + right.y, this->z + right.z);
		}

		Float3 operator-(const Float3& right)
		{
			return Float3(this->x - right.x, this->y - right.y, this->z - right.z);
		}

		Float3 operator-()
		{
			return Float3(-this->x, -this->y, -this->z);
		}

		bool operator==(const Float3& right)
		{
			return (this->x == right.x && this->y == right.y && this->z == right.z);
		}

		bool operator!=(const Float3& right)
		{
			return (this->x != right.x || this->y != right.y || this->z != right.z);
		}
	};

	void PrintVector(Float3& f);

	void Normalize(Float3& f);
	float Length(Float3& f);
	float LengthSquared(Float3& f);
	float Dot(Float3& f1, Float3& f2);
	Float3 Cross(Float3& f1, Float3& f2);
	//void Negate(Float3& f);
	//void Add(Float3& f1, Float3& f2);
	//void Sub(Float3& f1, Float3& f2);
	//void Div(Float3& f1, float f2);
	//void Mul(Float3& f1, float f2);

	Float3 Reflect(Float3& left, Float3& normal);
	//Float3 MagProduct(Float3& v, float f);
	Float3 Lerp(Float3& v, float f);
}
