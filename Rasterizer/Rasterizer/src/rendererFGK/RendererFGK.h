#pragma once

#include "../SpecificObjectFactory.h"
#include "../IRenderer.h"
#include "../Float2.h"
#include "../Int2.h"
#include "../Matrix4x4.h"
#include "../Ray.h"

class Camera;

namespace rendererFGK
{
	class RendererFGK :
		public IRenderer
	{
		friend class SpecificObjectFactory;
	public:

#pragma region Enum Public

		enum AntialiasingMode
		{
			NONE,
			ADAPTIVE
		};

#pragma endregion
	protected:

#pragma region Struct Protected

		struct AdaptiveRays
		{
			union
			{
				struct
				{
					Ray center;
					Ray tl;
					Ray tr;
					Ray br;
					Ray bl;
				};
				Ray tab[5];
			};

			AdaptiveRays()
			{
				ZeroMemory(tab, 5 * sizeof(Ray));
			}

			AdaptiveRays(const AdaptiveRays& c)
			{
				this->center = c.center;
				this->tl = c.tl;
				this->tr = c.tr;
				this->br = c.br;
				this->bl = c.bl;
			}

			AdaptiveRays(const Ray& center, const Ray& tl, const Ray& tr, const Ray& br, const Ray& bl)
			{
				this->center = center;
				this->tl = tl;
				this->tr = tr;
				this->br = br;
				this->bl = bl;
			}

			~AdaptiveRays()
			{

			}
		};

#pragma endregion

#pragma region Const

		static const int32_t NUM_THREADS = 8;

#pragma endregion

#pragma region Protected

		AntialiasingMode _aaMode;
		math::Float2 _halfPxSize;
		float _aaColorDistance;
		Color32 _clearColor;
		uint8_t _aaDepth;

		HANDLE _threadHandles[NUM_THREADS];
		SYNCHRONIZATION_BARRIER _barrier;

#pragma endregion

#pragma region Functions Protected

		RendererFGK(SystemSettings* settings);

		
		void DestroyThreads();
		static DWORD WINAPI ThreadFunc(_In_ LPVOID lpParameter);

		inline void ComputePixel(math::Int2 pos, Scene* scene, Camera* cam, float tanFovByTwo);
		inline math::Float2 GetViewSpacePosition(const math::Int2& pos);
		inline math::Int2 GetScreenSpacePosition(const math::Float3& pos);
		inline Ray CalculateRay(const math::Float3& px, float tanFovByTwo, float aspect, const math::Matrix4x4* vmInv, math::Float3* camOrigin);
		inline Ray RendererFGK::CalculateRayOrtho(const math::Float3& px, float aspect, const math::Matrix4x4* vmInv, math::Float3* camOrigin, math::Float3* camDirection);
		inline Color32 RaySample(Ray& ray, Scene* scene, const math::Float3 camOrigin, const math::Int2 ndcPos);
		inline Color32 RaySampleAdaptive(AdaptiveRays& rays, math::Float2 ssPixel, math::Float2 halfPxSize, Scene* scene, 
			const math::Matrix4x4* vmInv, math::Float3* camOrigin, const math::Int2 ndcPos, float tanFovByTwo, float aspect, int ctr);

#pragma endregion

	public:

#pragma region Functions Public

		~RendererFGK();

		void InitThreads();
		virtual void Draw(Scene* scene) override;

#pragma endregion

	};

}