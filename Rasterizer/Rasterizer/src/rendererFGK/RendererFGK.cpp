#include "RendererFGK.h"
#include "../Scene.h"
#include "../Camera.h"
#include "../Primitive.h"

#include <stack>

namespace rendererFGK
{
	RendererFGK::RendererFGK(SystemSettings* settings) :
		IRenderer(settings),
		_aaMode(AntialiasingMode::NONE),
		_aaColorDistance(0.2f),
		_clearColor(0xFFAAAAAA),
		_aaDepth(4)
	{
		uint16_t w = _bufferColor.GetWidth();
		uint16_t h = _bufferColor.GetHeight();
		_halfPxSize = math::Float2(0.5f / (float)w, 0.5f / (float)h);
	}

	RendererFGK::~RendererFGK()
	{
	}

	void RendererFGK::Draw(Scene * scene)
	{
		_bufferColor.Fill(0x00AAAAAA);
		_bufferDepth.Fill(FLT_MAX);

		// for each pixel, cast ray into scene
		Camera* cam = scene->GetCurrentCamera();

		uint16_t w = _bufferColor.GetWidth();
		uint16_t h = _bufferColor.GetHeight();

		float tanFovByTwo = tan(cam->GetFOVYRads() * 0.5f);
		float aspect = cam->GetAspectRatio();
		float nearPlane = cam->GetNearPlane();

		math::Float3 camOrigin = *cam->GetPosition();
		math::Float3 camDirection = *cam->GetDirection();

		for (uint16_t i = 0; i < h; ++i)
		{
			for (uint16_t j = 0; j < w; ++j)
			{
				math::Float3 ssPixel(GetViewSpacePosition(math::Int2(j, i)));
				Ray ray = CalculateRay(ssPixel, tanFovByTwo, aspect, cam->GetViewInvMatrix(), &camOrigin);

				if (_aaMode == AntialiasingMode::ADAPTIVE)
				{
					AdaptiveRays aCoords
					(
						ray,
						CalculateRay(math::Float3(ssPixel.x - _halfPxSize.x, ssPixel.y + _halfPxSize.y, 0.0f), tanFovByTwo, aspect, cam->GetViewInvMatrix(), &camOrigin),
						CalculateRay(math::Float3(ssPixel.x + _halfPxSize.x, ssPixel.y + _halfPxSize.y, 0.0f), tanFovByTwo, aspect, cam->GetViewInvMatrix(), &camOrigin),
						CalculateRay(math::Float3(ssPixel.x + _halfPxSize.x, ssPixel.y - _halfPxSize.y, 0.0f), tanFovByTwo, aspect, cam->GetViewInvMatrix(), &camOrigin),
						CalculateRay(math::Float3(ssPixel.x - _halfPxSize.x, ssPixel.y - _halfPxSize.y, 0.0f), tanFovByTwo, aspect, cam->GetViewInvMatrix(), &camOrigin)
					);

					_bufferColor.SetPixel(j, i, RaySampleAdaptive(aCoords, math::Float2(ssPixel.x, ssPixel.y), _halfPxSize, scene, cam->GetViewInvMatrix(), &camOrigin, math::Int2(j, i), tanFovByTwo, aspect, 0));
				}
				else
				{
					_bufferColor.SetPixel(j, i, RaySample(ray, scene, camOrigin, math::Int2(j, i)));
				}
			}
		}
	}

	math::Float2 RendererFGK::GetViewSpacePosition(const math::Int2 & pos)
	{
		return math::Float2(
			(float)(pos.x) / _bufferColor.GetWidth() * 2.0f - 1.0f + _halfPxSize.x,
			(float)(pos.y) / _bufferColor.GetHeight() * 2.0f - 1.0f + _halfPxSize.y
			);
	}

	math::Int2 RendererFGK::GetScreenSpacePosition(const math::Float3 & pos)
	{
		return math::Int2
			(
				(int32_t)(pos.x * (float)_bufferColor.GetWidth() * 0.5f + ((float)_bufferColor.GetWidth() * 0.5f)),
				(int32_t)(pos.y * (float)_bufferColor.GetHeight() * 0.5f + ((float)_bufferColor.GetHeight() * 0.5f))
			);
	}

	Ray RendererFGK::CalculateRay(const math::Float3& px, float tanFovByTwo, float aspect, const math::Matrix4x4* vmInv, math::Float3* camOrigin)
	{
		math::Float3 point = px * tanFovByTwo;
		point.x *= aspect;
		point.z = 1.0f;
		point = *vmInv * math::Float4(point);
		point = point - *camOrigin;
		math::Float3::Normalize(point);
		return Ray(*camOrigin, point);
	}

	inline Color32 RendererFGK::RaySample(Ray & ray, Scene * scene, const math::Float3 camOrigin, const math::Int2 ndcPos)
	{
		Color32 ret = _clearColor;
		std::vector<Primitive*>* prims = scene->GetPrimitives();
		float closestDist = FLT_MAX;
		Primitive* prim = nullptr;
		for (std::vector<Primitive*>::iterator it = prims->begin(); it != prims->end(); ++it)
		{
			RayHit hit = (*it)->CalcIntersect(ray);
			if (hit.hit)
			{
				float distanceToCamera = math::Float3::LengthSquared(hit.point - camOrigin);
				if (distanceToCamera <= closestDist)	// depth test
				{
					closestDist = distanceToCamera;
					prim = (*it);
				}
			}
		}

		if (prim != nullptr)
		{
			Material* mat = prim->GetMaterialPtr();
			if (mat != nullptr)
			{
				ret = *mat->GetColorDiffuse();
			}
			else
			{
				// no material - draw white
				ret.color = 0xFFFFFFFF;
			}
		}
		return ret;
	}

	Color32 RendererFGK::RaySampleAdaptive(AdaptiveRays& rays, math::Float2 ssPixel, math::Float2 halfPxSize, Scene* scene,
		const math::Matrix4x4* vmInv, math::Float3* camOrigin, const math::Int2 ndcPos, float tanFovByTwo, float aspect, int ctr)
	{
		// sample four corner rays
		Color32 cols[4];
		for (size_t i = 1; i < 5; ++i)
		{
			cols[i - 1] = RaySample(rays.tab[i], scene, *camOrigin, ndcPos);
		}

		// check recursion warunek
		if (ctr < _aaDepth)
		{
			// check corner ray colour
			for (size_t k = 0; k < 4; ++k)
			{
				float dist = 0.0f;
				for (size_t m = k + 1; m < k + 3; ++m)
				{
					size_t mCap = m % 4;
					dist += Color32::Distance(cols[k], cols[mCap]);
				}
				if (dist > _aaColorDistance)
				{
					// distance is bigger, so sample further this pixel
					//halfPxSize = halfPxSize * 0.5f;
					if (k == 0)
					{
						// tl
						cols[k] = RaySampleAdaptive
						(
							AdaptiveRays
							(
								CalculateRay(math::Float3(ssPixel.x - halfPxSize.x * 0.5f, ssPixel.y + halfPxSize.y * 0.5f, 0.0f), tanFovByTwo, aspect, vmInv, camOrigin),
								rays.tl,
								CalculateRay(math::Float3(ssPixel.x, ssPixel.y + halfPxSize.y, 0.0f), tanFovByTwo, aspect, vmInv, camOrigin),
								rays.center,
								CalculateRay(math::Float3(ssPixel.x - halfPxSize.x, ssPixel.y, 0.0f), tanFovByTwo, aspect, vmInv, camOrigin)
							),
							math::Float2(ssPixel.x - halfPxSize.x * 0.5f, ssPixel.y + halfPxSize.y * 0.5f),
							halfPxSize * 0.5f,
							scene,
							vmInv,
							camOrigin,
							ndcPos,
							tanFovByTwo,
							aspect,
							ctr + 1
						);
					}
					else if (k == 1)
					{
						// tr
						cols[k] = RaySampleAdaptive
						(
							AdaptiveRays
							(
								CalculateRay(math::Float3(ssPixel.x + halfPxSize.x * 0.5f, ssPixel.y + halfPxSize.y * 0.5f, 0.0f), tanFovByTwo, aspect, vmInv, camOrigin),
								CalculateRay(math::Float3(ssPixel.x, ssPixel.y + halfPxSize.y, 0.0f), tanFovByTwo, aspect, vmInv, camOrigin),
								rays.tr,
								CalculateRay(math::Float3(ssPixel.x + halfPxSize.x, ssPixel.y, 0.0f), tanFovByTwo, aspect, vmInv, camOrigin),
								rays.center
							),
							math::Float2(ssPixel.x + halfPxSize.x * 0.5f, ssPixel.y + halfPxSize.y * 0.5f),
							halfPxSize * 0.5f,
							scene,
							vmInv,
							camOrigin,
							ndcPos,
							tanFovByTwo,
							aspect,
							ctr + 1
						);
					}
					else if (k == 2)
					{
						// br
						cols[k] = RaySampleAdaptive
						(
							AdaptiveRays
							(
								CalculateRay(math::Float3(ssPixel.x + halfPxSize.x * 0.5f, ssPixel.y - halfPxSize.y * 0.5f, 0.0f), tanFovByTwo, aspect, vmInv, camOrigin),
								rays.center,
								CalculateRay(math::Float3(ssPixel.x + halfPxSize.x, ssPixel.y, 0.0f), tanFovByTwo, aspect, vmInv, camOrigin),
								rays.br,
								CalculateRay(math::Float3(ssPixel.x, ssPixel.y - halfPxSize.y, 0.0f), tanFovByTwo, aspect, vmInv, camOrigin)
							),
							math::Float2(ssPixel.x + halfPxSize.x * 0.5f, ssPixel.y - halfPxSize.y * 0.5f),
							halfPxSize * 0.5f,
							scene,
							vmInv,
							camOrigin,
							ndcPos,
							tanFovByTwo,
							aspect,
							ctr + 1
						);
					}
					else
					{
						// bl
						cols[k] = RaySampleAdaptive
						(
							AdaptiveRays
							(
								CalculateRay(math::Float3(ssPixel.x - halfPxSize.x * 0.5f, ssPixel.y - halfPxSize.y * 0.5f, 0.0f), tanFovByTwo, aspect, vmInv, camOrigin),
								CalculateRay(math::Float3(ssPixel.x - halfPxSize.x, ssPixel.y, 0.0f), tanFovByTwo, aspect, vmInv, camOrigin),
								rays.center,
								CalculateRay(math::Float3(ssPixel.x, ssPixel.y - halfPxSize.y, 0.0f), tanFovByTwo, aspect, vmInv, camOrigin),
								rays.bl
							),
							math::Float2(ssPixel.x + halfPxSize.x * 0.5f, ssPixel.y - halfPxSize.y * 0.5f),
							halfPxSize * 0.5f,
							scene,
							vmInv,
							camOrigin,
							ndcPos,
							tanFovByTwo,
							aspect,
							ctr + 1
						);
					}
				}
			}
 
			return Color32::AverageFour(cols);
		}
		else
		{
			return Color32::AverageFour(cols);
		}
	}

}