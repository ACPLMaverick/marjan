#pragma once

#include "../Triangle.h"
#include "../SpecificObjectFactory.h"

namespace rendererMav
{
	class TriangleMav : public Triangle
	{
		friend class SpecificObjectFactory;
	protected:
#pragma region Functions Protected

		TriangleMav(math::Float3 x, math::Float3 y, math::Float3 z, Color32 col = Color32());

#pragma endregion

	public:

#pragma region Functions Public

		virtual ~TriangleMav();

		virtual void Draw();

#pragma endregion
	};
}