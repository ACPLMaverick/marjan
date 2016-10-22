#include "TriangleFGK.h"


namespace rendererFGK
{

	TriangleFGK::TriangleFGK(math::Float3 x, math::Float3 y, math::Float3 z,
		math::Float2 ux, math::Float2 uy, math::Float2 uz,
		math::Float3 cx, math::Float3 cy, math::Float3 cz,
		Color32 col) :
		Triangle(x, y, z, ux, uy, uz, cx, cy, cz, col),
		_plane(x, y, z)
	{
	}


	TriangleFGK::~TriangleFGK()
	{
	}

	RayHit TriangleFGK::CalcIntersect(Ray & ray)
	{
		// plane check
		RayHit rayHit(_plane.CalcIntersect(ray));
		if (rayHit.hit)
		{
			// bounding box check
			math::Float3 bbMin
			(
				min(min(v1.x, v2.x), v3.x),
				min(min(v1.y, v2.y), v3.y),
				min(min(v1.z, v2.z), v3.z)
			);
			math::Float3 bbMax
			(
				max(max(v1.x, v2.x), v3.x),
				max(max(v1.y, v2.y), v3.y),
				max(max(v1.z, v2.z), v3.z)
			);

			if (
				rayHit.point >= bbMin &&
				rayHit.point <= bbMax
				)
			{
				// barycentric check

				// constant pre-calculation
				float dx2x1 = v2.x - v1.x;
				float dy2y1 = v2.y - v1.y;
				float dx3x2 = v3.x - v2.x;
				float dy3y2 = v3.y - v2.y;
				float dx1x3 = v1.x - v3.x;
				float dy1y3 = v1.y - v3.y;

				// baycentric coords data pre-calculation
				float bd00 = Float2Dot(dx2x1, dy2y1, dx2x1, dy2y1);
				float bd01 = Float2Dot(dx2x1, dy2y1, -dx1x3, -dy1y3);
				float bd11 = Float2Dot(-dx1x3, -dy1y3, -dx1x3, -dy1y3);
				float bdenom = 1.0f / (bd00 * bd11 - bd01 * bd01);

				// barycentric coords calculation
				float bv2x = (float)rayHit.point.x - (float)v1.x;
				float bv2y = (float)rayHit.point.y - (float)v1.y;
				float bd20 = Float2Dot(bv2x, bv2y, (float)dx2x1, (float)dy2y1);
				float bd21 = Float2Dot(bv2x, bv2y, -(float)dx1x3, -(float)dy1y3);

				rayHit.barycentric = math::Float3
				(
					(bd11 * bd20 - bd01 * bd21) * bdenom,
					(bd00 * bd21 - bd01 * bd20) * bdenom,
					0.0f
				);
				rayHit.barycentric.z = 1.0f - rayHit.barycentric.x - rayHit.barycentric.y;

				if (
					(rayHit.barycentric.x < 0.0f || rayHit.barycentric.x > 1.0f) ||
					(rayHit.barycentric.y < 0.0f || rayHit.barycentric.y > 1.0f) ||
					(rayHit.barycentric.z < 0.0f || rayHit.barycentric.z > 1.0f)
					)
				{
					rayHit.hit = false;
				}
				else
				{
					rayHit.hit = true;
				}
			}
			else
			{
				rayHit.hit = false;
			}
		}

		return rayHit;
	}

	void TriangleFGK::Draw()
	{
	}

}