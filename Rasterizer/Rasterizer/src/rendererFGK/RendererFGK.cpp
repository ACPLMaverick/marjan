#include "RendererFGK.h"
#include "../Scene.h"
#include "../Camera.h"
#include "../Ray.h"
#include "../Primitive.h"
#include "../Matrix4x4.h"

namespace rendererFGK
{
	RendererFGK::RendererFGK(SystemSettings* settings) :
		IRenderer(settings)
	{
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
		std::vector<Primitive*>* prims = scene->GetPrimitives();

		for (uint16_t i = 0; i < h; ++i)
		{
			for (uint16_t j = 0; j < w; ++j)
			{
				math::Float3 point(GetViewSpacePosition(math::Int2(j, i)));
				point *= tanFovByTwo;
				point.x *= aspect;
				point.z = 1.0f;
				point = *cam->GetViewInvMatrix() * math::Float4(point);
				point = point - camOrigin;
				math::Float3::Normalize(point);

				Ray ray(
					camOrigin,
					point
					);

				for (std::vector<Primitive*>::iterator it = prims->begin(); it != prims->end(); ++it)
				{
					RayHit hit = (*it)->CalcIntersect(ray);
					if (hit.hit)
					{
						float distanceToCamera = math::Float3::LengthSquared(hit.point - camOrigin);
						if (distanceToCamera < _bufferDepth.GetPixel(j, i))	// depth test
						{
							Material* mt = (*it)->GetMaterialPtr();
							_bufferColor.SetPixel(j, i, *mt->GetColorDiffuse());
							_bufferDepth.SetPixel(j, i, distanceToCamera);
						}
					}
				}
			}
		}
	}

	math::Float2 RendererFGK::GetViewSpacePosition(const math::Int2 & pos)
	{
		return math::Float2(
			(float)(pos.x) / _bufferColor.GetWidth() * 2.0f - 1.0f,
			(float)(pos.y) / _bufferColor.GetHeight() * 2.0f - 1.0f
			);
	}

	inline math::Int2 RendererFGK::GetScreenSpacePosition(const math::Float3 & pos)
	{
		return math::Int2
			(
				(int32_t)(pos.x * (float)_bufferColor.GetWidth() * 0.5f + ((float)_bufferColor.GetWidth() * 0.5f)),
				(int32_t)(pos.y * (float)_bufferColor.GetHeight() * 0.5f + ((float)_bufferColor.GetHeight() * 0.5f))
			);
	}

}