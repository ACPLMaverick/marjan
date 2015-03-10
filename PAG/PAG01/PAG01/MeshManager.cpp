#include "MeshManager.h"


MeshManager::MeshManager()
{
}


MeshManager::~MeshManager()
{
}

bool MeshManager::Initialize(GLuint programID)
{
	// load textures - test
	Texture* m_texture = new Texture();
	if (!m_texture->Initialize(&TEST_DIFFUSE)) return false;
	textures.push_back(m_texture);

	if(!Load3DS(FILEPATH_FIXED)) return false;

	/*
	// load meshes - test
	Mesh* m_mesh = new Mesh();
	if (!m_mesh->Initialize(programID, NULL)) return false;
	m_mesh->SetTexture(m_texture);
	m_mesh->Transform(&glm::vec3(0.0f, 1.0f, 0.0f), &glm::vec3(0.0f, 0.0f, 0.0f), &glm::vec3(1.0f, 1.0f, 1.0f));

	Mesh* test = new Mesh();
	if (!test->Initialize(programID, NULL)) return false;
	test->SetTexture(m_texture);
	m_mesh->AddChild(test);
	test->Transform(&glm::vec3(-3.0f, 0.0f, 0.0f), &glm::vec3(0.0f, 0.0f, 0.0f), &glm::vec3(1.0f, 1.0f, 1.0f));

	meshes.push_back(m_mesh);
	*/

	return true;
}

void MeshManager::Shutdown()
{
	if (meshes.size() > 0)
	{
		for (vector<Mesh*>::iterator it = meshes.begin(); it != meshes.end(); ++it)
		{
			(*it)->Shutdown();
			delete (*it);
		}
	}
	if (textures.size() > 0)
	{
		for (vector<Texture*>::iterator it = textures.begin(); it != textures.end(); ++it)
		{
			(*it)->Shutdown();
			delete (*it);
		}
	}
}

void MeshManager::Draw(glm::mat4* projectionMatrix, glm::mat4* viewMatrix, glm::vec3* eyeVector, GLuint eyeVectorID, Light* light)
{
	for (vector<Mesh*>::iterator it = meshes.begin(); it != meshes.end(); ++it)
	{
		(*it)->Draw(projectionMatrix, viewMatrix, eyeVector, eyeVectorID, light);
	}
}

bool MeshManager::Load3DS(string filePath)
{
	ifstream stream = ifstream(filePath.c_str(), ios::in | ios::binary);
	string file, buffer;
	while (!stream.eof())
	{
		stream >> buffer;
		file += buffer;
	}
	stream.close();

	

	return true;
}

void MeshManager::AddMesh(Mesh* mesh)
{
	meshes.push_back(mesh);
}

Mesh* MeshManager::GetMesh(unsigned int i)
{
	if (meshes.size() > 0) return meshes.at(i);
	else return NULL;
}