#include "Float3.h"

namespace math
{
	bool Float3::EqualsEpsilon(Float3 & right, float epsilon)
	{
		return *this >= (right - epsilon) && *this <= (right + epsilon);
	}

	bool Float3::GreaterEpsilon(Float3 & right, float epsilon)
	{
		return *this > (right - epsilon);
	}

	bool Float3::SmallerEpsilon(Float3 & right, float epsilon)
	{
		return *this < (right + epsilon);
	}

	bool Float3::GreaterEqualsEpsilon(Float3 & right, float epsilon)
	{
		return *this >= (right - epsilon);
	}

	bool Float3::SmallerEqualsEpsilon(Float3 & right, float epsilon)
	{
		return *this <= (right + epsilon);
	}

	void Float3::Normalize(Float3& f)
	{
		float n = Length(f);
		if (n >= 0.000001f)
		{
			f = f / n;
		}
		else
		{
			f.x = f.y = f.z = 0.0f;
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

	float Float3::Dot(const Float3 & f1, const Float3 & f2)
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

	Float3 Float3::Lerp(Float3& a, Float3 & b, float f)
	{
		return Float3(
			FloatLerp(a.x, b.x, f),
			FloatLerp(a.y, b.y, f),
			FloatLerp(a.z, b.z, f)
		);
	}
}