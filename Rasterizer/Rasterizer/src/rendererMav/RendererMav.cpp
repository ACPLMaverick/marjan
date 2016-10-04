#include "RendererMav.h"
#include "../Scene.h"

namespace rendererMav
{
	RendererMav::RendererMav(SystemSettings * settings) :
		IRenderer(settings)
	{
	}


	RendererMav::~RendererMav()
	{
	}

	void RendererMav::Draw(Scene * scene)
	{
		_bufferColor.Fill(0x00CCCCCC);
		_bufferDepth.Fill(-FLT_MAX);
		scene->Draw();
	}

	Buffer<float>* RendererMav::GetDepthBuffer()
	{
		return &_bufferDepth;
	}
}