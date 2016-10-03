#pragma once

#include "stdafx.h"
#include "SystemSettings.h"

class Triangle;
class IRenderer;

class SpecificObjectFactory
{
protected:

#pragma region Protected

#pragma endregion

#pragma region Functions Protected

	SpecificObjectFactory();

#pragma endregion

public:

#pragma region Functions Public

	virtual ~SpecificObjectFactory();

#pragma endregion

#pragma region Functions Public Static

	static IRenderer* GetRenderer(SystemSettings* ss);
	static Triangle* GetTriangle(Float3 x, Float3 y, Float3 z, Color32 col = Color32());

#pragma endregion
};
