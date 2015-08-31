#include "Component.h"


Component::Component(SimObject* obj)
{
	m_obj = obj;
}

Component::Component(const Component*)
{

}

Component::~Component()
{
}

SimObject* Component::GetMySimObject()
{
	return m_obj;
}