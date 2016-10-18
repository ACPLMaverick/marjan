#pragma once

#include "../SpecificObjectFactory.h"
#include "../IRenderer.h"
#include "../Float2.h"
#include "../Int2.h"

namespace rendererFGK
{
	class RendererFGK :
		public IRenderer
	{
		friend class SpecificObjectFactory;
	protected:

#pragma region Const

#pragma endregion

#pragma region Protected

		RendererFGK(SystemSettings* settings);

		inline math::Float2 GetViewSpacePosition(const math::Int2& pos);
		inline math::Int2 GetScreenSpacePosition(const math::Float3& pos);

#pragma endregion

	public:

#pragma region Functions Public

		~RendererFGK();

		virtual void Draw(Scene* scene) override;

#pragma endregion

	};

}