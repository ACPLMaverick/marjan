#include "SpecificObjectFactory.h"

#include "rendererMav/RendererMav.h"
#include "rendererMav/TriangleMav.h"

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

Triangle * SpecificObjectFactory::GetTriangle(math::Float3 x, math::Float3 y, math::Float3 z, Color32 col)
{
#ifdef RENDERER_FGK

#endif // RENDERER_FGK

#ifdef RENDERER_MAV

	return new rendererMav::TriangleMav(x, y, z, col);

#endif // RENDERER_MAV

#ifdef RENDERER_MAJSTER

#endif // RENDERER_MAJSTER
}