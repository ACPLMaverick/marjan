#pragma once

namespace math
{
	struct Float2
	{
		union
		{
			float tab[2];
			struct { float u; float v; };
			struct { float x; float y; };
		};
		Float2();
		Float2(float nx, float ny);
		~Float2();

		Float2 operator*(const float right) const;
		Float2 operator+(const Float2& right) const;
	};
}