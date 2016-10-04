#include "TriangleMav.h"
#include "RendererMav.h"
#include "../System.h"

namespace rendererMav
{
	TriangleMav::TriangleMav(math::Float3 x, math::Float3 y, math::Float3 z,
		math::Float3 ux, math::Float3 uy, math::Float3 uz,
		math::Float3 cx, math::Float3 cy, math::Float3 cz,
		Color32 col) :
		Triangle(x, y, z, ux, uy, uz, cx, cy, cz, col)
	{
	}

	TriangleMav::~TriangleMav()
	{
	}

	void TriangleMav::Draw()
	{
		Buffer<Color32>* buf = System::GetInstance()->GetRenderer()->GetColorBuffer();
		Buffer<float>* depth = ((RendererMav*)System::GetInstance()->GetRenderer())->GetDepthBuffer();
		int32_t v1x, v1y, v2x, v2y, v3x, v3y;
		v1x = (int32_t)ConvertFromScreenToBuffer(v1.x, buf->GetWidth());
		v1y = (int32_t)ConvertFromScreenToBuffer(v1.y, buf->GetHeight());
		v2x = (int32_t)ConvertFromScreenToBuffer(v2.x, buf->GetWidth());
		v2y = (int32_t)ConvertFromScreenToBuffer(v2.y, buf->GetHeight());
		v3x = (int32_t)ConvertFromScreenToBuffer(v3.x, buf->GetWidth());
		v3y = (int32_t)ConvertFromScreenToBuffer(v3.y, buf->GetHeight());

		// triangle bounding box
		int32_t minX = (min(min(v1x, v2x), v3x));
		int32_t minY = (min(min(v1y, v2y), v3y));
		int32_t maxX = (max(max(v1x, v2x), v3x));
		int32_t maxY = (max(max(v1y, v2y), v3y));

		// screen clipping
		minX = max(minX, 0);
		minY = max(minY, 0);
		maxX = min(maxX, (int32_t)buf->GetWidth() - 1);
		maxY = min(maxY, (int32_t)buf->GetHeight() - 1);

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
					float cDepth = depth->GetPixel(j, i);
					float mDepth = v1.z * bv + v2.z * bw + v3.z * bu;
					if (mDepth > cDepth)
					{
						// color interpolation
						Color32 finalColor = c1 * bv + c2 * bw + c3 * bu;
						finalColor *= col;

						// alpha blend
						Color32 cColor = buf->GetPixel(j, i);
						finalColor = Color32::LerpNoAlpha(cColor, finalColor, ((float)finalColor.a / 255.0f));

						// write output color to buffer
						buf->SetPixel(j, i, finalColor);

						// write z to depth buffer
						depth->SetPixel(j, i, mDepth);
					}
				}
			}
		}
	}

	uint16_t TriangleMav::ConvertFromScreenToBuffer(float point, uint16_t maxValue)
	{
		return (uint16_t)(point * (float)maxValue * 0.5f + ((float)maxValue * 0.5f));
	}

}