#include "Matrix4x4.h"

namespace math
{
	Matrix4x4::Matrix4x4()
	{
		ZeroMemory(tabf, sizeof(float) * 16);
		a00 = a11 = a22 = a33 = 1.0f;
	}

	Matrix4x4::Matrix4x4(
		const Float4& r1,
		const Float4& r2,
		const Float4& r3,
		const Float4& r4
	)
	{
		row1 = r1;
		row2 = r2;
		row3 = r3;
		row4 = r4;
	}

	Matrix4x4::~Matrix4x4()
	{

	}

	Matrix4x4 & Matrix4x4::operator=(const Matrix4x4 & right)
	{
		Matrix4x4 ret;
		for (size_t i = 0; i < 4; ++i)
		{
			ret.tabf[i] = right.tabf[i];
		}
		return ret;
	}

	Matrix4x4 Matrix4x4::operator*(const Matrix4x4& right)
	{
		for (size_t i = 0; i < 4; ++i)
		{
			tabf[i].x = Float4::Dot(row1, Float4(right.a00, right.a10, right.a20, right.a30));
			tabf[i].y = Float4::Dot(row1, Float4(right.a01, right.a11, right.a21, right.a31));
			tabf[i].z = Float4::Dot(row1, Float4(right.a02, right.a12, right.a22, right.a32));
			tabf[i].w = Float4::Dot(row1, Float4(right.a03, right.a13, right.a23, right.a33));
		}

		return *this;
	}

	Matrix4x4& Matrix4x4::operator*(const float right)
	{
		for (size_t i = 0; i < 4; ++i)
		{
			tabf[i] *= right;
		}
		return *this;
	}

	Matrix4x4& Matrix4x4::operator*=(const Matrix4x4& right)
	{
		*this = *this * right;
		return *this;
	}

	Matrix4x4& Matrix4x4::operator*=(const float right)
	{
		*this = *this * right;
		return *this;
	}

	Matrix4x4& Matrix4x4::operator+(const Matrix4x4& right)
	{
		for (size_t i = 0; i < 4; ++i)
		{
			tabf[i] = tabf[i] + right.tabf[i];
		}
		return *this;
	}

	Matrix4x4& Matrix4x4::operator+=(const Matrix4x4& right)
	{
		*this = *this + right;
		return *this;
	}

	Float4& Matrix4x4::operator[](const size_t ind)
	{
		return tabf[ind];
	}
	
	Float4& operator*(Float4& left, const Matrix4x4& right)
	{
		left = Float4(right.a00, right.a10, right.a20, right.a30) * left.x +
			Float4(right.a01, right.a11, right.a21, right.a31) * left.y +
			Float4(right.a02, right.a12, right.a22, right.a32) * left.z +
			Float4(right.a03, right.a13, right.a23, right.a33) * left.w;
		return left;
	}

	std::ostream& operator<<(std::ostream& ost, const Matrix4x4& m)
	{
		for (int i = 0; i < 4; ++i)
		{
			ost << m.tabf[i];
		}
		return ost;
	}

	void Matrix4x4::Identity(Matrix4x4 * out)
	{
		*out = Matrix4x4();
	}

	void Matrix4x4::Translation(const Float3 * trans, Matrix4x4 * out)
	{

	}

	void Matrix4x4::Scale(const Float3 * scale, Matrix4x4 * out)
	{

	}

	void Matrix4x4::Rotate(const Float3 * rotationXYZdeg, Matrix4x4 * out)
	{

	}

	void Matrix4x4::Transpose(const Matrix4x4 * in, Matrix4x4 * out)
	{

	}

	void Matrix4x4::Inverse(const Matrix4x4 * in, Matrix4x4 * out)
	{

	}

	void Matrix4x4::LookAt(const Float3 * cameraPos, const Float3 * cameraTarget, const Float3 * cameraUp, Matrix4x4 * out)
	{

	}

	void Matrix4x4::Perspective(const float fovAngle, const float aspectRatio, const float nearPlane, const float farPlane, Matrix4x4 * out)
	{

	}
}
