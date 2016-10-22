#pragma once

#include "../Triangle.h"
#include "../Plane.h"

namespace rendererFGK
{

	class TriangleFGK :
		public Triangle
	{

#pragma region Protected

		Plane _plane;

#pragma endregion
	
	public:

#pragma region Functions Public

		TriangleFGK(math::Float3 x, math::Float3 y, math::Float3 z,
			math::Float2 ux, math::Float2 uy, math::Float2 uz,
			math::Float3 cx = math::Float3(1.0f, 1.0f, 1.0f), math::Float3 cy = math::Float3(1.0f, 1.0f, 1.0f), math::Float3 cz = math::Float3(1.0f, 1.0f, 1.0f),
			Color32 col = Color32());
		virtual ~TriangleFGK();

		virtual RayHit CalcIntersect(Ray& ray) override;
		virtual void Draw() override;

#pragma endregion

	};

}
