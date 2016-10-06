#pragma once

#include "../stdafx.h"
#include "../Buffer.h"
#include "../Color32.h"
#include "../Float3.h"
#include "../Int2.h"
#include "../Matrix4x4.h"

class GraphicsDevice
{
protected:

#pragma region Protected Structs

	struct VertexInput
	{
		math::Float3 Position;
		math::Float3 Color;
		math::Float3 Uv;
	};

	struct VertexOutput
	{
		math::Float3 Position;
		math::Float3 WorldPosition;
		math::Float3 Color;
		math::Float3 Uv;
	};

	struct PixelInput
	{
		math::Float3 WorldPosition;
		math::Float3 Uv;
		math::Int2 Position;
		Color32 Color;
	};

	struct PixelOutput
	{
		Color32 color;
	};
	
#pragma endregion

#pragma region Protected const

	const float _fltInv255 = 1.0f / 255.0f;

#pragma endregion

#pragma region Protected

	Buffer<Color32>* _bufferColor;
	Buffer<float>* _bufferDepth;

	const math::Float3* _vb;
	const math::Float3* _cb;
	const math::Float3* _ub;

	const math::Matrix4x4* _wvpMat;
	const math::Matrix4x4* _wMat;
	const math::Matrix4x4* _wInvTransMat;

#pragma endregion

#pragma region Functions Protected

	virtual inline int32_t ConvertFromScreenToBuffer(float point, uint16_t maxValue);

	virtual inline void VertexShader
		(
			const VertexInput& in,
			VertexOutput& out
		);

	virtual inline void Rasterizer
		(
			const VertexOutput& in1,
			const VertexOutput& in2,
			const VertexOutput& in3
		);

	virtual inline void PixelShader
		(
			const PixelInput& in,
			Color32& out
		);

#pragma endregion

public:

#pragma region Functions Public

	GraphicsDevice();
	~GraphicsDevice();

	void Initialize(Buffer<Color32>* cb, Buffer<float>* db);
	void Shutdown();

	void Draw(size_t triangleNum);
	void DrawIndexed(size_t triangleNum, const uint16_t* ib);

	void SetVertexBuffer(const math::Float3* buf);
	void SetColorBuffer(const math::Float3* buf);
	void SetUVBuffer(const math::Float3* buf);

	void SetWorldViewProjMatrix(const math::Matrix4x4* m);
	void SetWorldMatrix(const math::Matrix4x4* m);
	void SetWorldInverseTransposeMatrix(const math::Matrix4x4* m);

#pragma endregion
};

