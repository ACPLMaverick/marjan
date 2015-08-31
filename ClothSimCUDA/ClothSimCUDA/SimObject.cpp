#include "SimObject.h"


SimObject::SimObject()
{
}

SimObject::SimObject(const SimObject*)
{
}


SimObject::~SimObject()
{
}

unsigned int SimObject::Initialize()
{
	return CS_ERR_NONE;
}

unsigned int SimObject::Shutdown()
{
	return CS_ERR_NONE;
}

unsigned int SimObject::Update()
{
	return CS_ERR_NONE;
}

unsigned int SimObject::Draw()
{
	return CS_ERR_NONE;
}