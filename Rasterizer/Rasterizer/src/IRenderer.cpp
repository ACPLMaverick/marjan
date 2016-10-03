#include "IRenderer.h"



IRenderer::IRenderer(SystemSettings* settings) :
	_bufferColor(settings->GetDisplayWidth(), settings->GetDisplayHeight()),
	_bufferDepth(settings->GetDisplayWidth(), settings->GetDisplayHeight())
{
}


IRenderer::~IRenderer()
{
}
