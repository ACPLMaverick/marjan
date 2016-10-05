#include "Float3.h"

namespace math
{
	void Float3::PrintVector(Float3& f)
	{
		std::cout << "Float3(" << f.x << ", " << f.y << ", " << f.z << ")" << std::endl;
	}

	void Float3::Normalize(Float3& f)
	{
		float n = Length(f);
		if (n != 0)
		{
			f = f / n;
		}
	}

	float Float3::Length(Float3 & f)
	{
		return sqrt(LengthSquared(f));
	}

	float Float3::LengthSquared(Float3 & f)
	{
		return (f.x * f.x) + (f.y * f.y) + (f.z * f.z);
	}

	float Float3::Dot(Float3 & f1, Float3 & f2)
	{
		return (f1.x * f2.x + f1.y * f2.y + f1.z * f2.z);
	}

	Float3 Float3::Cross(Float3 & f1, Float3 & f2)
	{
		return Float3(f1.y * f2.z - f1.z * f2.y, f1.z * f2.x - f1.x * f2.z, f1.x * f2.y - f1.y * f2.x);
	}

	Float3 Float3::Reflect(Float3 & left, Float3 & normal)
	{
		return left - (normal * 2.0f * Dot(left, normal));
	}

	Float3 Float3::Lerp(Float3 & v, float f)
	{
		Float3 out;
		out.x = v.x + f * (out.x + v.x);
		out.y = v.y + f * (out.y + v.y);
		out.z = v.z + f * (out.z + v.z);
		return out;
	}
}