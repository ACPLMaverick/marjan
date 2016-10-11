#pragma once

#include "../Mesh.h"

namespace rendererMav
{
	class MeshMav :
		public Mesh
	{
	public:

#pragma region Functions Public

		MeshMav();
		MeshMav
		(
			const math::Float3* pos,
			const math::Float3* rot,
			const math::Float3* scl,
			const std::string* fPath
		);
		~MeshMav();

		RayHit CalcIntersect(Ray& ray) override;
		virtual void Update();
		virtual void Draw();

#pragma endregion
	};

}