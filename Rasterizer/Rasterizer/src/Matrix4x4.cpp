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

	Matrix4x4 Matrix4x4::operator=(const Matrix4x4 & right) const
	{
		Matrix4x4 ret;
		memcpy(&ret, &right, sizeof(float) * 16);
		//for (size_t i = 0; i < 4; ++i)
		//{
		//	ret.tabf[i] = right.tabf[i];
		//}
		return ret;
	}

	Matrix4x4 Matrix4x4::operator*(const Matrix4x4& right) const
	{
		Matrix4x4 ret;
		for (size_t i = 0; i < 4; ++i)
		{
			ret.tabf[i].x = Float4::Dot(tabf[i], Float4(right.a00, right.a10, right.a20, right.a30));
			ret.tabf[i].y = Float4::Dot(tabf[i], Float4(right.a01, right.a11, right.a21, right.a31));
			ret.tabf[i].z = Float4::Dot(tabf[i], Float4(right.a02, right.a12, right.a22, right.a32));
			ret.tabf[i].w = Float4::Dot(tabf[i], Float4(right.a03, right.a13, right.a23, right.a33));
		}

		return ret;
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
	
	Float4& operator*(const Matrix4x4& left, Float4& right)
	{
		right = Float4(left.a00, left.a10, left.a20, left.a30) * right.x +
			Float4(left.a01, left.a11, left.a21, left.a31) * right.y +
			Float4(left.a02, left.a12, left.a22, left.a32) * right.z +
			Float4(left.a03, left.a13, left.a23, left.a33) * right.w;
		return right;
	}

	std::ostream& operator<<(std::ostream& ost, const Matrix4x4& m)
	{
		for (size_t i = 0; i < 4; ++i)
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
		Identity(out);
		out->a03 = trans->x;
		out->a13 = trans->y;
		out->a23 = trans->z;
	}

	void Matrix4x4::Scale(const Float3 * scale, Matrix4x4 * out)
	{
		Identity(out);
		out->a00 = scale->x;
		out->a11 = scale->y;
		out->a22 = scale->z;
	}

	void Matrix4x4::Rotation(const Float3 * rotationXYZdeg, Matrix4x4 * out)
	{
		Float3 rad = Float3(
			DegToRad(rotationXYZdeg->x),
			DegToRad(rotationXYZdeg->y),
			DegToRad(rotationXYZdeg->z)
			);

		Matrix4x4 x = Matrix4x4(
			Float4(1.0f, 0.0f, 0.0f, 0.0f),
			Float4(0.0f, cos(rad.x), -sin(rad.x), 0.0f),
			Float4(0.0f, sin(rad.x), cos(rad.x), 0.0f),
			Float4(0.0f, 0.0f, 0.0f, 1.0f)
			);
		Matrix4x4 y = Matrix4x4(
			Float4(cos(rad.y), 0.0f, sin(rad.y), 0.0f),
			Float4(0.0f, 1.0f, 0.0f, 0.0f),
			Float4(-sin(rad.y), 0.0f, cos(rad.y), 0.0f),
			Float4(0.0f, 0.0f, 0.0f, 1.0f)
			);
		Matrix4x4 z = Matrix4x4(
			Float4(cos(rad.z), -sin(rad.z), 0.0f, 0.0f),
			Float4(sin(rad.z), cos(rad.z), 0.0f, 0.0f),
			Float4(0.0f, 0.0f, 1.0f, 0.0f),
			Float4(0.0f, 0.0f, 0.0f, 1.0f)
			);

		*out = x * y * z;
	}

	void Matrix4x4::Transpose(const Matrix4x4 * in, Matrix4x4 * out)
	{
		out->a00 = in->a00;
		out->a01 = in->a10;
		out->a02 = in->a20;
		out->a03 = in->a30;
		out->a10 = in->a01;
		out->a11 = in->a11;
		out->a12 = in->a21;
		out->a13 = in->a31;
		out->a20 = in->a02;
		out->a21 = in->a12;
		out->a22 = in->a22;
		out->a23 = in->a32;
		out->a30 = in->a03;
		out->a31 = in->a13;
		out->a32 = in->a23;
		out->a33 = in->a33;
	}

	void Matrix4x4::Inverse(const Matrix4x4 * in, Matrix4x4 * out)
	{
		float inv[16], det;
		int i;

		inv[0] = in->tab[5] * in->tab[10] * in->tab[15] -
			in->tab[5] * in->tab[11] * in->tab[14] -
			in->tab[9] * in->tab[6] * in->tab[15] +
			in->tab[9] * in->tab[7] * in->tab[14] +
			in->tab[13] * in->tab[6] * in->tab[11] -
			in->tab[13] * in->tab[7] * in->tab[10];

		inv[4] = -in->tab[4] * in->tab[10] * in->tab[15] +
			in->tab[4] * in->tab[11] * in->tab[14] +
			in->tab[8] * in->tab[6] * in->tab[15] -
			in->tab[8] * in->tab[7] * in->tab[14] -
			in->tab[12] * in->tab[6] * in->tab[11] +
			in->tab[12] * in->tab[7] * in->tab[10];

		inv[8] = in->tab[4] * in->tab[9] * in->tab[15] -
			in->tab[4] * in->tab[11] * in->tab[13] -
			in->tab[8] * in->tab[5] * in->tab[15] +
			in->tab[8] * in->tab[7] * in->tab[13] +
			in->tab[12] * in->tab[5] * in->tab[11] -
			in->tab[12] * in->tab[7] * in->tab[9];

		inv[12] = -in->tab[4] * in->tab[9] * in->tab[14] +
			in->tab[4] * in->tab[10] * in->tab[13] +
			in->tab[8] * in->tab[5] * in->tab[14] -
			in->tab[8] * in->tab[6] * in->tab[13] -
			in->tab[12] * in->tab[5] * in->tab[10] +
			in->tab[12] * in->tab[6] * in->tab[9];

		inv[1] = -in->tab[1] * in->tab[10] * in->tab[15] +
			in->tab[1] * in->tab[11] * in->tab[14] +
			in->tab[9] * in->tab[2] * in->tab[15] -
			in->tab[9] * in->tab[3] * in->tab[14] -
			in->tab[13] * in->tab[2] * in->tab[11] +
			in->tab[13] * in->tab[3] * in->tab[10];

		inv[5] = in->tab[0] * in->tab[10] * in->tab[15] -
			in->tab[0] * in->tab[11] * in->tab[14] -
			in->tab[8] * in->tab[2] * in->tab[15] +
			in->tab[8] * in->tab[3] * in->tab[14] +
			in->tab[12] * in->tab[2] * in->tab[11] -
			in->tab[12] * in->tab[3] * in->tab[10];

		inv[9] = -in->tab[0] * in->tab[9] * in->tab[15] +
			in->tab[0] * in->tab[11] * in->tab[13] +
			in->tab[8] * in->tab[1] * in->tab[15] -
			in->tab[8] * in->tab[3] * in->tab[13] -
			in->tab[12] * in->tab[1] * in->tab[11] +
			in->tab[12] * in->tab[3] * in->tab[9];

		inv[13] = in->tab[0] * in->tab[9] * in->tab[14] -
			in->tab[0] * in->tab[10] * in->tab[13] -
			in->tab[8] * in->tab[1] * in->tab[14] +
			in->tab[8] * in->tab[2] * in->tab[13] +
			in->tab[12] * in->tab[1] * in->tab[10] -
			in->tab[12] * in->tab[2] * in->tab[9];

		inv[2] = in->tab[1] * in->tab[6] * in->tab[15] -
			in->tab[1] * in->tab[7] * in->tab[14] -
			in->tab[5] * in->tab[2] * in->tab[15] +
			in->tab[5] * in->tab[3] * in->tab[14] +
			in->tab[13] * in->tab[2] * in->tab[7] -
			in->tab[13] * in->tab[3] * in->tab[6];

		inv[6] = -in->tab[0] * in->tab[6] * in->tab[15] +
			in->tab[0] * in->tab[7] * in->tab[14] +
			in->tab[4] * in->tab[2] * in->tab[15] -
			in->tab[4] * in->tab[3] * in->tab[14] -
			in->tab[12] * in->tab[2] * in->tab[7] +
			in->tab[12] * in->tab[3] * in->tab[6];

		inv[10] = in->tab[0] * in->tab[5] * in->tab[15] -
			in->tab[0] * in->tab[7] * in->tab[13] -
			in->tab[4] * in->tab[1] * in->tab[15] +
			in->tab[4] * in->tab[3] * in->tab[13] +
			in->tab[12] * in->tab[1] * in->tab[7] -
			in->tab[12] * in->tab[3] * in->tab[5];

		inv[14] = -in->tab[0] * in->tab[5] * in->tab[14] +
			in->tab[0] * in->tab[6] * in->tab[13] +
			in->tab[4] * in->tab[1] * in->tab[14] -
			in->tab[4] * in->tab[2] * in->tab[13] -
			in->tab[12] * in->tab[1] * in->tab[6] +
			in->tab[12] * in->tab[2] * in->tab[5];

		inv[3] = -in->tab[1] * in->tab[6] * in->tab[11] +
			in->tab[1] * in->tab[7] * in->tab[10] +
			in->tab[5] * in->tab[2] * in->tab[11] -
			in->tab[5] * in->tab[3] * in->tab[10] -
			in->tab[9] * in->tab[2] * in->tab[7] +
			in->tab[9] * in->tab[3] * in->tab[6];

		inv[7] = in->tab[0] * in->tab[6] * in->tab[11] -
			in->tab[0] * in->tab[7] * in->tab[10] -
			in->tab[4] * in->tab[2] * in->tab[11] +
			in->tab[4] * in->tab[3] * in->tab[10] +
			in->tab[8] * in->tab[2] * in->tab[7] -
			in->tab[8] * in->tab[3] * in->tab[6];

		inv[11] = -in->tab[0] * in->tab[5] * in->tab[11] +
			in->tab[0] * in->tab[7] * in->tab[9] +
			in->tab[4] * in->tab[1] * in->tab[11] -
			in->tab[4] * in->tab[3] * in->tab[9] -
			in->tab[8] * in->tab[1] * in->tab[7] +
			in->tab[8] * in->tab[3] * in->tab[5];

		inv[15] = in->tab[0] * in->tab[5] * in->tab[10] -
			in->tab[0] * in->tab[6] * in->tab[9] -
			in->tab[4] * in->tab[1] * in->tab[10] +
			in->tab[4] * in->tab[2] * in->tab[9] +
			in->tab[8] * in->tab[1] * in->tab[6] -
			in->tab[8] * in->tab[2] * in->tab[5];

		det = in->tab[0] * inv[0] + in->tab[1] * inv[4] + in->tab[2] * inv[8] + in->tab[3] * inv[12];

		if (det == 0)
			return;

		det = 1.0 / det;

		for (i = 0; i < 16; i++)
			out->tab[i] = inv[i] * det;
	}

	void Matrix4x4::LookAt(const Float3 * cameraPos, const Float3 * cameraTarget, const Float3 * cameraUp, Matrix4x4 * out)
	{
		Matrix4x4 eyeTrans, eyeRot;
		eyeTrans.a03 = -cameraPos->x;
		eyeTrans.a13 = -cameraPos->y;
		eyeTrans.a23 = -cameraPos->z;

		Float3 forward = *cameraTarget - *cameraPos;
		Float3 up = *cameraUp;
		Float3 right = Float3::Cross(forward, up);

		eyeRot.a00 = right.x;
		eyeRot.a01 = right.y;
		eyeRot.a02 = right.z;
		eyeRot.a10 = up.x;
		eyeRot.a11 = up.y;
		eyeRot.a12 = up.z;
		eyeRot.a20 = -forward.x;
		eyeRot.a21 = -forward.y;
		eyeRot.a22 = -forward.z;

		*out = eyeTrans;
	}

	void Matrix4x4::Perspective(const float fovAngle, const float aspectRatio, const float nearPlane, const float farPlane, Matrix4x4 * out)
	{
		float dNear = farPlane - nearPlane;
		float fADiv2 = fovAngle * 0.5f;
		float aRDiv2 = aspectRatio * 0.5f;
		Identity(out);
		out->a00 = cos(aRDiv2) / sin(aRDiv2);
		out->a11 = cos(fADiv2) / sin(fADiv2);
		out->a22 = -(farPlane + nearPlane) / dNear;
		out->a23 = -2.0f * (farPlane * nearPlane) / dNear;
		out->a32 = -1.0f;
	}
}