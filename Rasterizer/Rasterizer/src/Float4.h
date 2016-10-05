#pragma once

#include <Windows.h>
#include <iostream>

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

		Float4();
		Float4(float nx, float ny, float nz, float nw);
		~Float4();

		Float4 operator*(const Float4& right) const;
		Float4 operator*(float scalar) const;
		Float4& operator*=(float scalar);
		Float4 operator/(float scalar) const;
		Float4 operator+(const Float4& right) const;
		Float4 operator-(const Float4& right) const;
		Float4 operator-() const;
		bool operator==(const Float4& right) const;
		bool operator!=(const Float4& right) const;
		float operator[](const size_t ind);
		friend std::ostream& operator<<(std::ostream& ost, const Float4& flt);

		static void Normalize(Float4& f);
		static float Length(const Float4& f);
		static float LengthSquared(const Float4& f);
		static float Dot(const Float4& f1, const Float4& f2);
		static Float4 Cross(const Float4& f1, const Float4& f2);

		static Float4 Reflect(const Float4& left, const Float4& normal);
		static Float4 Lerp(const Float4 & a, const Float4& b, float f);
	};
}