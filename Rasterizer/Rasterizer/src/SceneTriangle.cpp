#include "SceneTriangle.h"
#include "Camera.h"
#include "Triangle.h"
#include "SpecificObjectFactory.h"


SceneTriangle::SceneTriangle()
{
}


SceneTriangle::~SceneTriangle()
{
}

void SceneTriangle::InitializeScene()
{
	_cameras.push_back(new Camera());
	_primitives.push_back(SpecificObjectFactory::GetTriangle(
		math::Float3(-0.5f, -0.5f, 0.0f),
		math::Float3(0.0f, 0.8f, 0.0f),
		math::Float3(0.5f, -0.5f, 0.0f),

		math::Float3(0.0f, 0.0f, 0.0f),
		math::Float3(0.0f, 1.0f, 0.0f),
		math::Float3(1.0f, 1.0f, 0.0f),

		math::Float3(1.0f, 0.0f, 0.0f),
		math::Float3(0.0f, 1.0f, 0.0f),
		math::Float3(0.0f, 0.0f, 1.0f),

		Color32(0xFFFFFFFF)
	));
	_primitives.push_back(SpecificObjectFactory::GetTriangle(
		math::Float3(-0.3f, 0.8f, 1.0f),
		math::Float3(0.5f, 0.8f, 1.0f),
		math::Float3(0.2f, -0.5f, -1.0f),

		math::Float3(0.0f, 0.0f, 0.0f),
		math::Float3(0.0f, 1.0f, 0.0f),
		math::Float3(1.0f, 1.0f, 0.0f),

		math::Float3(1.0f, 1.0f, 0.0f),
		math::Float3(0.0f, 1.0f, 1.0f),
		math::Float3(1.0f, 0.0f, 1.0f),

		Color32(0xAFFFFFFF)
	));
}