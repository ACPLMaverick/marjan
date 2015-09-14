#include "Mesh.h"


Mesh::Mesh(SimObject* obj) : Component(obj)
{
	m_texID = nullptr;
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



void Mesh::SetTextureID(TextureID* id)
{
	m_texID = id;
}

TextureID* Mesh::GetTextureID()
{
	return m_texID;
}