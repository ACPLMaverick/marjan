#pragma once

#include "../IRenderer.h"
#include "../SpecificObjectFactory.h"

namespace rendererMav
{
	class RendererMav :
		public IRenderer
	{
		friend class SpecificObjectFactory;
	protected:

#pragma region Functions Protected

		RendererMav(SystemSettings* settings);

#pragma endregion

	public:

#pragma region Functions Public

		~RendererMav();

		virtual void Draw(Scene* scene);

#pragma endregion
	};

}
