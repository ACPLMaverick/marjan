#include "Float4.h"
#include "Float3.h"
#include "Matrix4x4.h"

namespace math
{
	Float4::Float4()
	{
		x = 0.0f;
		y = 0.0f;
		z = 0.0f;
		w = 0.0f;
	}

	Float4::Float4(float nx, float ny, float nz, float nw)
	{
		x = nx;
		y = ny;
		z = nz;
		w = nw;
	}

	Float4::~Float4()
	{

	}

	Float4 Float4::operator*(const Float4& right) const
	{
		return Float4(this->x * right.x, this->y * right.y, this->z * right.z, this->w * right.w);
	}

	Float4 Float4::operator*(float scalar) const
	{
		return Float4(this->x * scalar, this->y * scalar, this->z * scalar, this->w * scalar);
	}

	Float4& Float4::operator*=(float scalar)
	{
		this->x * scalar; this->y * scalar; this->z * scalar; this->w * scalar;
		return *this;
	}

	Float4 Float4::operator/(float scalar) const
	{
		float inverse = 1.0f / scalar;

		return *this * inverse;
	}

	Float4 Float4::operator+(const Float4& right) const
	{
		return Float4(this->x + right.x, this->y + right.y, this->z + right.z, this->w + right.w);
	}

	Float4 Float4::operator-(const Float4& right) const
	{
		return Float4(this->x - right.x, this->y - right.y, this->z - right.z, this->w - right.w);
	}

	Float4 Float4::operator-() const
	{
		return Float4(-this->x, -this->y, -this->z, -this->w);
	}

	bool Float4::operator==(const Float4& right) const
	{
		return (this->x == right.x && this->y == right.y && this->z == right.z && this->w == right.w);
	}

	bool Float4::operator!=(const Float4& right) const
	{
		return (this->x != right.x || this->y != right.y || this->z != right.z || this->w != right.w);
	}

	float Float4::operator[](const size_t ind)
	{
		return tab[ind];
	}

	std::ostream & operator<<(std::ostream & ost, const Float4 & flt)
	{
		ost << "(" << flt.x << ", " << flt.y << ", " << flt.z << ", " << flt.w << ")" << std::endl;
		return ost;
	}

	void Float4::Normalize(Float4 & f)
	{
		float n = Length(f);
		if (n >= 0.000001f)
		{
			f = f / n;
		}
		else
		{
			f.x = f.y = f.z = f.w = 0.0f;
		}
	}

	float Float4::Length(const Float4 & f)
	{
		return sqrt(LengthSquared(f));
	}

	float Float4::LengthSquared(const Float4 & f)
	{
		return (f.x * f.x) + (f.y * f.y) + (f.z * f.z) + (f.w * f.w);
	}

	float Float4::Dot(const Float4 & f1, const Float4 & f2)
	{
		return (f1.x * f2.x + f1.y * f2.y + f1.z * f2.z + f1.w * f2.w);
	}

	Float4 Float4::Cross(const Float4 & f1, const Float4 & f2)
	{
		Float3 f1c = Float3(f1.x, f1.y, f1.z);
		Float3 f2c = Float3(f2.x, f2.y, f2.z);
		Float3 c = Float3::Cross(f1c, f2c);
		return Float4(c.x, c.y, c.z, f1.w);
	}

	Float4 Float4::Reflect(const Float4 & left, const Float4 & normal)
	{
		return left - (normal * 2.0f * Dot(left, normal));
	}

	Float4 Float4::Lerp(const Float4 & a, const Float4 & b, float f)
	{
		return Float4(
			FloatLerp(a.x, b.x, f),
			FloatLerp(a.y, b.y, f),
			FloatLerp(a.z, b.z, f),
			FloatLerp(a.w, b.w, f)
		);
	}

}