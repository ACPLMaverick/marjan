#include "Scene.h"


Scene::Scene()
{
}

Scene::Scene(const Scene*)
{
}


Scene::~Scene()
{
}

unsigned int Scene::Initialize()
{
	return CS_ERR_NONE;
}

unsigned int Scene::Shutdown()
{
	return CS_ERR_NONE;
}

unsigned int Scene::Update()
{
	return CS_ERR_NONE;
}

unsigned int Scene::Draw()
{
	return CS_ERR_NONE;
}