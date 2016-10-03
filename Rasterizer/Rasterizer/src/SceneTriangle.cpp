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
		Float3(-0.5f, -0.5f, 0.0f), 
		Float3(0.0f, 0.8f, 0.0f), 
		Float3(0.5f, -0.5f, 0.0f)
	));
}