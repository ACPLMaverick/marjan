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
		math::Float3 camOrigin = *cam->GetPosition();
		math::Float3 camDirection = *cam->GetDirection();
		std::vector<Primitive*>* prims = scene->GetPrimitives();

		math::Matrix4x4 matInv;
		math::Matrix4x4::Inverse(cam->GetProjMatrix(), &matInv);

		for (uint16_t i = 0; i < h; ++i)
		{
			for (uint16_t j = 0; j < w; ++j)
			{
				math::Float3 origin(GetViewSpacePosition(math::Int2(j, i)));
				math::Float4 originP(origin.x, origin.y, 0.0f, 1.0f);
				origin = matInv * originP;



				math::Float3 dir = origin - camOrigin;
				dir = dir + camDirection;
				math::Float3::Normalize(dir);

				Ray ray(
					camOrigin + origin,
					dir
					);

				for (std::vector<Primitive*>::iterator it = prims->begin(); it != prims->end(); ++it)
				{
					RayHit hit = (*it)->CalcIntersect(ray);
					if (hit.hit)
					{
						Material* mt = (*it)->GetMaterialPtr();
						//math::Int2 sPos = GetScreenSpacePosition(hit.point);
						_bufferColor.SetPixel(j, i, *mt->GetColorDiffuse());
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