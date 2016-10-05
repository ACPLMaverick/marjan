#include "GraphicsDevice.h"



GraphicsDevice::GraphicsDevice()
{
}

GraphicsDevice::~GraphicsDevice()
{
}

void GraphicsDevice::Initialize(Buffer<Color32>* cb, Buffer<float>* db)
{
	_bufferColor = cb;
	_bufferDepth = db;
}

void GraphicsDevice::Shutdown()
{
}

void GraphicsDevice::Draw(size_t triangleNum)
{
	for (size_t i = 0; i < triangleNum; i += 3)
	{
		VertexInput vinput1, vinput2, vinput3;
		VertexOutput voutput1, voutput2, voutput3;
		vinput1.Position = _vb[i];
		vinput2.Position = _vb[i + 1];
		vinput3.Position = _vb[i + 2];
		vinput1.Color = _cb[i];
		vinput2.Color = _cb[i + 1];
		vinput3.Color = _cb[i + 2];
		vinput1.Uv = _ub[i];
		vinput2.Uv = _ub[i + 1];
		vinput3.Uv = _ub[i + 2];

		VertexShader(vinput1, voutput1);
		VertexShader(vinput2, voutput2);
		VertexShader(vinput3, voutput3);

		Rasterizer(voutput1, voutput2, voutput3);
	}
}

void GraphicsDevice::DrawIndexed(size_t triangleNum)
{

}

void GraphicsDevice::SetVertexBuffer(math::Float3 * buf)
{
	_vb = buf;
}

void GraphicsDevice::SetColorBuffer(math::Float3 * buf)
{
	_cb = buf;
}

void GraphicsDevice::SetUVBuffer(math::Float3 * buf)
{
	_ub = buf;
}

uint16_t GraphicsDevice::ConvertFromScreenToBuffer(float point, uint16_t maxValue)
{
	return (uint16_t)(point * (float)maxValue * 0.5f + ((float)maxValue * 0.5f));
}


void GraphicsDevice::VertexShader(const VertexInput & in, VertexOutput & out)
{
	out.Position = in.Position;
	out.WorldPosition = in.Position;
	out.Color = in.Color;
	out.Uv = in.Uv;
}

void GraphicsDevice::Rasterizer(const VertexOutput& in1, const VertexOutput& in2, const VertexOutput& in3)
{
	int32_t v1x, v1y, v2x, v2y, v3x, v3y;
	v1x = (int32_t)ConvertFromScreenToBuffer(in1.Position.x, _bufferColor->GetWidth());
	v1y = (int32_t)ConvertFromScreenToBuffer(in1.Position.y, _bufferColor->GetHeight());
	v2x = (int32_t)ConvertFromScreenToBuffer(in2.Position.x, _bufferColor->GetWidth());
	v2y = (int32_t)ConvertFromScreenToBuffer(in2.Position.y, _bufferColor->GetHeight());
	v3x = (int32_t)ConvertFromScreenToBuffer(in3.Position.x, _bufferColor->GetWidth());
	v3y = (int32_t)ConvertFromScreenToBuffer(in3.Position.y, _bufferColor->GetHeight());

	// triangle bounding box
	int32_t minX = (min(min(v1x, v2x), v3x));
	int32_t minY = (min(min(v1y, v2y), v3y));
	int32_t maxX = (max(max(v1x, v2x), v3x));
	int32_t maxY = (max(max(v1y, v2y), v3y));

	// screen clipping
	minX = max(minX, 0);
	minY = max(minY, 0);
	maxX = min(maxX, (int32_t)_bufferColor->GetWidth() - 1);
	maxY = min(maxY, (int32_t)_bufferColor->GetHeight() - 1);

	// constant pre-calculation
	int32_t dx2x1 = v2x - v1x;
	int32_t dy2y1 = v2y - v1y;
	int32_t dx3x2 = v3x - v2x;
	int32_t dy3y2 = v3y - v2y;
	int32_t dx1x3 = v1x - v3x;
	int32_t dy1y3 = v1y - v3y;

	// top-left rule booleans
	bool e21isTopLeft = dy2y1 > 0 || (dy2y1 == 0 && dx2x1 > 0);
	bool e32isTopLeft = dy3y2 > 0 || (dy3y2 == 0 && dx3x2 > 0);
	bool e12isTopLeft = dy1y3 > 0 || (dy1y3 == 0 && dx1x3 > 0);

	// baycentric coords data pre-calculation
	float bd00 = Float2Dot((float)dx2x1, (float)dy2y1, (float)dx2x1, (float)dy2y1);
	float bd01 = Float2Dot((float)dx2x1, (float)dy2y1, -(float)dx1x3, -(float)dy1y3);
	float bd11 = Float2Dot(-(float)dx1x3, -(float)dy1y3, -(float)dx1x3, -(float)dy1y3);
	float bdenom = 1.0f / (bd00 * bd11 - bd01 * bd01);

	for (int32_t i = minY; i <= maxY; ++i)
	{
		for (int32_t j = minX; j <= maxX; ++j)
		{
			int32_t e21edgeEquation = (dx2x1 * (i - v1y) - dy2y1 * (j - v1x));
			int32_t e32edgeEquation = (dx3x2 * (i - v2y) - dy3y2 * (j - v2x));
			int32_t e12edgeEquation = (dx1x3 * (i - v3y) - dy1y3 * (j - v3x));
			if (
				((e21edgeEquation < 0) || (e21isTopLeft && e21edgeEquation <= 0)) &&
				((e32edgeEquation < 0) || (e32isTopLeft && e32edgeEquation <= 0)) &&
				((e12edgeEquation < 0) || (e12isTopLeft && e12edgeEquation <= 0))
				)
			{
				// barycentric coords calculation
				float bv2x = (float)j - (float)v1x;
				float bv2y = (float)i - (float)v1y;
				float bd20 = Float2Dot(bv2x, bv2y, (float)dx2x1, (float)dy2y1);
				float bd21 = Float2Dot(bv2x, bv2y, -(float)dx1x3, -(float)dy1y3);

				float bw = (bd11 * bd20 - bd01 * bd21) * bdenom;
				float bu = (bd00 * bd21 - bd01 * bd20) * bdenom;
				float bv = 1.0f - bw - bu;

				// z-buffer clipping check
				// depth interpolation
				float cDepth = _bufferDepth->GetPixel(j, i);
				float mDepth = in1.Position.z * bv + in2.Position.z * bw + in3.Position.z * bu;
				if (mDepth > cDepth)
				{
					// write z to depth buffer
					_bufferDepth->SetPixel(j, i, mDepth);

					PixelInput pi;
					pi.Position = math::Int2(j, i);
					pi.WorldPosition = in1.WorldPosition * bv + in2.WorldPosition * bw + in3.WorldPosition * bu;
					pi.Color = in1.Color * bv + in2.Color * bw + in3.Color * bu;
					pi.Uv = in1.Uv * bv + in2.Uv * bw + in3.Uv * bu;
					Color32 col;
					PixelShader(pi, col);

					// write output color to buffer
					_bufferColor->SetPixel(j, i, col);
				}
			}
		}
	}
}

// save depth here

void GraphicsDevice::PixelShader(const PixelInput & in, Color32 & out)
{
	// alpha blend
	//Color32 cColor = _bufferColor->GetPixel(in.Position.x, in.Position.y);
	//out = Color32::LerpNoAlpha(cColor, in.Color, ((float)in.Color.a * _fltInv255));
	out = in.Color;
}
