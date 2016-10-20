#include "IRenderer.h"



IRenderer::IRenderer(SystemSettings* settings) :
	_bufferColor(settings->GetDisplayWidth() * 0.1f, settings->GetDisplayHeight() * 0.1f),
	_bufferDepth(settings->GetDisplayWidth() * 0.1f, settings->GetDisplayHeight() * 0.1f)
{
}


IRenderer::~IRenderer()
{
}
