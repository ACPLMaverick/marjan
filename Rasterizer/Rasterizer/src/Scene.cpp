#include "stdafx.h"
#include "Scene.h"
#include "Primitive.h"
#include "Camera.h"

Scene::Scene()
{
}


Scene::~Scene()
{
}

void Scene::Initialize(uint32_t uID, std::string * name)
{
	_name = *name;
	_uID = uID;

	InitializeScene();
}

void Scene::Shutdown()
{
	for (std::vector<Primitive*>::iterator it = _primitives.begin(); it != _primitives.end(); ++it)
	{
		delete *it;
	}

	for (std::vector<Camera*>::iterator it = _cameras.begin(); it != _cameras.end(); ++it)
	{
		delete *it;
	}

	_primitives.clear();
	_cameras.clear();
}

void Scene::Update()
{
	if (_flagToAddPrimitive)
	{
		_flagToAddPrimitive = false;
		for (std::vector<Primitive*>::iterator it = _primitivesToAdd.begin(); it != _primitivesToAdd.end(); ++it)
		{
			_primitives.push_back(*it);
		}
		_primitivesToAdd.clear();
	}
	if (_flagToRemovePrimitive)
	{
		_flagToRemovePrimitive = false;
		for (std::vector<std::vector<Primitive*>::iterator>::iterator it = _primitivesToRemove.begin(); it != _primitivesToRemove.end(); ++it)
		{
			_primitives.erase(*it);
		}
		_primitivesToRemove.clear();
	}

	for (std::vector<Primitive*>::iterator it = _primitives.begin(); it != _primitives.end(); ++it)
	{
		(*it)->Update();
	}

	for (std::vector<Camera*>::iterator it = _cameras.begin(); it != _cameras.end(); ++it)
	{
		(*it)->Update();
	}
}

void Scene::Draw()
{
	for (std::vector<Primitive*>::iterator it = _primitives.begin(); it != _primitives.end(); ++it)
	{
		(*it)->Draw();
	}
}

Camera * const Scene::GetCamera(uint32_t uid)
{
	for (std::vector<Camera*>::iterator it = _cameras.begin(); it != _cameras.end(); ++it)
	{
		if ((*it)->GetUID() == uid)
		{
			return *it;
		}
	}

	return nullptr;
}

Camera * const Scene::GetCamera(std::string * name)
{
	for (std::vector<Camera*>::iterator it = _cameras.begin(); it != _cameras.end(); ++it)
	{
		if (*(*it)->GetName() == *name)
		{
			return *it;
		}
	}

	return nullptr;
}

void Scene::AddPrimitive(Primitive * const Primitive)
{
	_primitivesToAdd.push_back(Primitive);
	_flagToAddPrimitive = true;
}

void Scene::AddCamera(Camera * const camera)
{
	_cameras.push_back(camera);
}

Camera * const Scene::RemoveCamera(uint32_t uid)
{
	for (std::vector<Camera*>::iterator it = _cameras.begin(); it != _cameras.end(); ++it)
	{
		if ((*it)->GetUID() == uid)
		{
			Camera* tmp = *it;
			_cameras.erase(it);
			return *it;
		}
	}

	return nullptr;
}

Camera * const Scene::RemoveCamera(std::string * name)
{
	for (std::vector<Camera*>::iterator it = _cameras.begin(); it != _cameras.end(); ++it)
	{
		if (*(*it)->GetName() == *name)
		{
			Camera* tmp = *it;
			_cameras.erase(it);
			return *it;
		}
	}

	return nullptr;
}
