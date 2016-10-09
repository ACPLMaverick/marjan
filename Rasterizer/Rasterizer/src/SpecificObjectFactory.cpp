#include "SpecificObjectFactory.h"

#include "rendererMav/RendererMav.h"
#include "rendererMav/TriangleMav.h"
#include "rendererMav/MeshMav.h"

SpecificObjectFactory::SpecificObjectFactory()
{
}


SpecificObjectFactory::~SpecificObjectFactory()
{
}

IRenderer * SpecificObjectFactory::GetRenderer(SystemSettings* ss)
{
#ifdef RENDERER_FGK

#endif // RENDERER_FGK

#ifdef RENDERER_MAV

	return new rendererMav::RendererMav(ss);

#endif // RENDERER_MAV

#ifdef RENDERER_MAJSTER

#endif // RENDERER_MAJSTER
}

Triangle * SpecificObjectFactory::GetTriangle(math::Float3 x, math::Float3 y, math::Float3 z,
	math::Float2 ux, math::Float2 uy, math::Float2 uz,
	math::Float3 cx, math::Float3 cy, math::Float3 cz,
	Color32 col)
{
#ifdef RENDERER_FGK

#endif // RENDERER_FGK

#ifdef RENDERER_MAV

	return new rendererMav::TriangleMav(x, y, z, ux, uy, uz, cx, cy, cz, col);

#endif // RENDERER_MAV

#ifdef RENDERER_MAJSTER

#endif // RENDERER_MAJSTER
}

Mesh * SpecificObjectFactory::GetMesh(const math::Float3 * pos, const math::Float3 * rot, const math::Float3 * scl, const std::string * fPath)
{
#ifdef RENDERER_FGK

#endif // RENDERER_FGK

#ifdef RENDERER_MAV

	return new rendererMav::MeshMav(pos, rot, scl, fPath);

#endif // RENDERER_MAV

#ifdef RENDERER_MAJSTER

#endif // RENDERER_MAJSTER
}
