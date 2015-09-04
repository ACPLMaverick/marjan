#include "Scene.h"


Scene::Scene(string n)
{
	m_name = n;
	m_currentObjectID = 0;
	m_currentCameraID = 0;
}

Scene::Scene(const Scene*)
{
}


Scene::~Scene()
{
}

unsigned int Scene::Shutdown()
{
	unsigned int err = CS_ERR_NONE;

	for (vector<SimObject*>::iterator it = m_objects.begin(); it != m_objects.end(); ++it)
	{
		err = (*it)->Shutdown();

		if (err != CS_ERR_NONE)
			return err;

		delete (*it);
	}
	m_objects.clear();

	for (vector<Camera*>::iterator it = m_cameras.begin(); it != m_cameras.end(); ++it)
	{
		err = (*it)->Shutdown();

		if (err != CS_ERR_NONE)
			return err;

		delete (*it);
	}
	m_cameras.clear();

	return CS_ERR_NONE;
}

unsigned int Scene::Update()
{
	unsigned int err = CS_ERR_NONE;

	for (vector<SimObject*>::iterator it = m_objects.begin(); it != m_objects.end(); ++it)
	{
		err = (*it)->Update();

		if (err != CS_ERR_NONE)
			return err;
	}

	return CS_ERR_NONE;
}

unsigned int Scene::Draw()
{
	unsigned int err;

	for (vector<SimObject*>::iterator it = m_objects.begin(); it != m_objects.end(); ++it)
	{
		err = (*it)->Draw();

		if (err != CS_ERR_NONE)
			return err;
	}

	return CS_ERR_NONE;
}



unsigned int Scene::GetCurrentObjectID()
{
	return m_currentObjectID;
}

unsigned int Scene::GetCurrentCameraID()
{
	return m_currentCameraID;
}



void Scene::AddObject(SimObject* obj)
{
	m_objects.push_back(obj);
}

void Scene::AddCamera(Camera* cam)
{
	m_cameras.push_back(cam);
}



void Scene::RemoveObject(SimObject* ptr)
{
	for (vector<SimObject*>::iterator it = m_objects.begin(); it != m_objects.end(); ++it)
	{
		if (ptr == (*it))
		{
			m_objects.erase(it);
			break;
		}
	}
}

void Scene::RemoveCamera(Camera* ptr)
{
	for (vector<Camera*>::iterator it = m_cameras.begin(); it != m_cameras.end(); ++it)
	{
		if (ptr == (*it))
		{
			m_cameras.erase(it);
			break;
		}
	}
}

void Scene::RemoveObject(unsigned int which)
{
	int ctr = 0;
	for (vector<SimObject*>::iterator it = m_objects.begin(); it != m_objects.end(); ++it)
	{
		if (ctr == which)
		{
			m_objects.erase(it);
			break;
		}
	}
}

void Scene::RemoveCamera(unsigned int which)
{
	int ctr = 0;
	for (vector<Camera*>::iterator it = m_cameras.begin(); it != m_cameras.end(); ++it)
	{
		if (ctr == which)
		{
			m_cameras.erase(it);
			break;
		}
	}
}



SimObject* Scene::GetObject()
{
	return m_objects.at(m_currentObjectID);
}

Camera* Scene::GetCamera()
{
	return m_cameras.at(m_currentCameraID);
}

SimObject* Scene::GetObject(unsigned int which)
{
	if (m_objects.size() < which)
		return nullptr;
	return m_objects.at(which);
}

Camera* Scene::GetCamera(unsigned int which)
{
	if (m_cameras.size() < which)
		return nullptr;
	return m_cameras.at(which);
}
