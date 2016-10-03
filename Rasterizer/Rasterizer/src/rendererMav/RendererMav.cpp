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
		_bufferColor.Fill(0x00FF0000);
		scene->Draw();
	}
}