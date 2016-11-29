#pragma once

#include <Windows.h>
#include <iostream>
#include "Color32.h"
#include "Float4.h"
#include "Float2.h"

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

		Float3(const Float4& g)
		{
			this->x = g.x;
			this->y = g.y;
			this->z = g.z;
		}

		Float3(const Float2& g)
		{
			this->x = g.x;
			this->y = g.y;
			this->z = 0.0f;
		}

		Float3 operator-() const
		{
			return Float3(-x, -y, -z);
		}

		Float3 operator*(const Float3& right) const
		{
			return Float3(this->x * right.x, this->y * right.y, this->z * right.z);
		}

		Float3 operator/(const Float3& right) const
		{
			return Float3(this->x / right.x, this->y / right.y, this->z / right.z);
		}

		Float3& operator/=(const Float3& right)
		{
			*this = *this / right;
			return *this;
		}

		Float3 operator*(float scalar) const
		{
			return Float3(this->x * scalar, this->y * scalar, this->z * scalar);
		}

		Float3& operator*=(float scalar)
		{
			*this = *this * scalar;
			return *this;
		}

		Float3 operator/(float scalar) const
		{
			float inverse = 1.0f / scalar;

			return Float3(this->x * inverse, this->y * inverse, this->z * inverse);
		}

		Float3& operator/=(float scalar)
		{
			*this = *this / scalar;
			return *this;
		}

		friend Float3 operator/(Float3& left, Float3& right)
		{
			return Float3(left.x / right.x, left.y / right.y, left.z / right.z);
		}

		friend Float3 operator/(float scalar, Float3& right)
		{
			return Float3(scalar / right.x, scalar / right.y, scalar / right.z);
		}

		Float3 operator+(const Float3& right) const
		{
			return Float3(this->x + right.x, this->y + right.y, this->z + right.z);
		}

		Float3 operator+(const float right) const
		{
			return Float3(this->x + right, this->y + right, this->z + right);
		}

		Float3 operator-(const float right) const
		{
			return Float3(this->x - right, this->y - right, this->z - right);
		}

		Float3 operator-(const Float3& right) const
		{
			return Float3(this->x - right.x, this->y - right.y, this->z - right.z);
		}

		Float3 operator+(const Float2& right) const
		{
			return Float3(this->x + right.x, this->y + right.y, this->z);
		}

		Float3 operator-(const Float2& right) const
		{
			return Float3(this->x - right.x, this->y - right.y, this->z);
		}

		Float3 operator-()
		{
			return Float3(-this->x, -this->y, -this->z);
		}

		bool operator==(const Float3& right)
		{
			return (this->x == right.x && this->y == right.y && this->z == right.z);
		}

		bool operator>(const Float3& right)
		{
			return (this->x > right.x && this->y > right.y && this->z > right.z);
		}

		bool operator<(const Float3& right)
		{
			return (this->x < right.x && this->y < right.y && this->z < right.z);
		}

		bool operator>=(const Float3& right)
		{
			return (this->x >= right.x && this->y >= right.y && this->z >= right.z);
		}

		bool operator<=(const Float3& right)
		{
			return (this->x <= right.x && this->y <= right.y && this->z <= right.z);
		}

		bool operator!=(const Float3& right)
		{
			return (this->x != right.x || this->y != right.y || this->z != right.z);
		}

		float operator[](const size_t ind)
		{
			return tab[ind];
		}

		friend std::ostream& operator<<(std::ostream& ost, const Float3 flt)
		{
			ost << "(" << flt.x << ", " << flt.y << ", " << flt.z << ")" << std::endl;
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

		bool EqualsEpsilon(const Float3& right, const float epsilon);
		bool GreaterEpsilon(const Float3& right, const float epsilon);
		bool SmallerEpsilon(const Float3& right, const float epsilon);
		bool GreaterEqualsEpsilon(const Float3& right, const float epsilon);
		bool SmallerEqualsEpsilon(const Float3& right, const float epsilon);
		
		static void Normalize(Float3& f);
		static float Length(const Float3& f);
		static float LengthSquared(const Float3& f);
		static float Dot(const Float3& f1, const Float3& f2);
		static Float3 SqrtComponentWise(const Float3& f);
		static Float3 Cross(const Float3& f1, const Float3& f2);
		static Float3 Reflect(const Float3& left, const Float3& normal);
		static Float3 Refract(const Float3& dir, const Float3& normal, const float coeff);
		static Float3 Lerp(const Float3& a, const Float3 & b, float f);
	};
}
