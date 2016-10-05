#pragma once

#include <Windows.h>
#include <iostream>
#include "Color32.h"

namespace math
{
	struct Float3
	{
		union
		{
			float tab[3];
			struct { float x, y, z; };
		};

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

		Float3 operator*(float scalar) const
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

		float operator[](const size_t ind)
		{
			return tab[ind];
		}

		operator Color32() const
		{
			Color32 data;

			data.a = 0xFF;
			float xc = Clamp(x, 0.0f, 1.0f);
			float yc = Clamp(y, 0.0f, 1.0f);
			float zc = Clamp(z, 0.0f, 1.0f);
			xc *= 255.0f;
			yc *= 255.0f;
			zc *= 255.0f;
			data.r = (uint8_t)(xc);
			data.g = (uint8_t)(yc);
			data.b = (uint8_t)(zc);

			return data;
		}

		static void PrintVector(Float3& f);
		
		static void Normalize(Float3& f);
		static float Length(Float3& f);
		static float LengthSquared(Float3& f);
		static float Dot(Float3& f1, Float3& f2);
		static Float3 Cross(Float3& f1, Float3& f2);
		
		static Float3 Reflect(Float3& left, Float3& normal);
		static Float3 Lerp(Float3& v, float f);
	};
}
