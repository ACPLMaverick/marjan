#include "Mesh.h"


Mesh::Mesh(SimObject* obj) : Component(obj)
{

}

Mesh::Mesh(const Mesh* m) : Component(m)
{
}


Mesh::~Mesh()
{
}

unsigned int Mesh::Update()
{
	return CS_ERR_NONE;
}
