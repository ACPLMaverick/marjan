#pragma once

#include <Windows.h>
#include "Float4.h"

namespace math
{
	struct Float4
	{
		union
		{
			float tab[4];
			struct { float x; float y; float z; float w; };
			struct { float r; float g; float b; float a; };
		};

		Float4()
		{
			x = 0.0f;
			y = 0.0f;
			z = 0.0f;
			w = 0.0f;
		}

		Float4(float nx, float ny, float nz, float nw)
		{
			x = nx;
			y = ny;
			z = nz;
			w = nw;
		}

		~Float4()
		{

		}

		Float4 operator*(const Float4& right) const
		{
			return Float4(this->x * right.x, this->y * right.y, this->z * right.z, this->w * right.w);
		}

		Float4 operator*(float scalar) const
		{
			return Float4(this->x * scalar, this->y * scalar, this->z * scalar, this->w * scalar);
		}

		Float4 operator/(float scalar) const
		{
			float inverse = 1.0f / scalar;

			return *this * inverse;
		}

		Float4 operator+(const Float4& right) const
		{
			return Float4(this->x + right.x, this->y + right.y, this->z + right.z, this->w + right.w);
		}

		Float4 operator-(const Float4& right) const
		{
			return Float4(this->x - right.x, this->y - right.y, this->z - right.z, this->w - right.w);
		}

		Float4 operator-() const
		{
			return Float4(-this->x, -this->y, -this->z, -this->w);
		}

		bool operator==(const Float4& right) const
		{
			return (this->x == right.x && this->y == right.y && this->z == right.z && this->w == right.w);
		}

		bool operator!=(const Float4& right) const
		{
			return (this->x != right.x || this->y != right.y || this->z != right.z || this->w != right.w);
		}

		float operator[](const size_t ind)
		{
			return tab[ind];
		}

		static void PrintVector(Float4& f);

		static void Normalize(Float4& f);
		static float Length(Float4& f);
		static float LengthSquared(Float4& f);
		static float Dot(Float4& f1, Float4& f2);
		static Float4 Cross(Float4& f1, Float4& f2);

		static Float4 Reflect(Float4& left, Float4& normal);
		static Float4 Lerp(Float4& v, float f);
	};
}