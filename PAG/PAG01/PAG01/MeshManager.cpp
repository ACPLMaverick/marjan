#include "MeshManager.h"


MeshManager::MeshManager()
{
}


MeshManager::~MeshManager()
{
}

bool MeshManager::Initialize(GLuint programID)
{
	m_programID = programID;
	// load textures - test
	Texture* m_texture = new Texture();
	if (!m_texture->Initialize(&TEST_DIFFUSE)) return false;
	textures.push_back(m_texture);

	if(!Load3DS(&FILEPATH_FIXED)) return false;

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

bool MeshManager::Load3DS(const string* filePath)
{
	printf("Loading 3DS...\n");
	string name;
	FILE* myFile;
	VertexData* data = new VertexData;
	list<GLfloat> vertices, uvs, normals;
	list<unsigned int> indices;
	// DONT FORGET ABOUT COLORS!!!oneoneone
	unsigned short chunkID;
	unsigned int chunkLength;
	unsigned char oneByte;
	unsigned short quantity;
	unsigned short quantityPolygons;
	unsigned short faceFlags;

	char tempChar = 'a';
	float tempFloat;
	unsigned short tempIndex;

	if ((fopen_s(&myFile, (*filePath).c_str(), "rb")) != 0)
	{
		return false;
	}

	while (ftell(myFile) < _filelength(_fileno(myFile)))
	{
		fread(&chunkID, 2, 1, myFile);
		fread(&chunkLength, 4, 1, myFile);

		switch (chunkID)
		{
			case ID_MAIN_CHUNK:
				oneByte = 5;
				break;
			case ID_EDITOR_CHUNK:
				oneByte = 5;
				break;
			case ID_OBJECT_BLOCK:

				oneByte = 5;
				for (int i = 0; i < 20 && tempChar != '\0'; i++)
				{
					fread(&tempChar, 1, 1, myFile);
					name.push_back(tempChar);
				}
				break;
			case ID_VERTICES_LIST:

				fread(&quantity, sizeof(unsigned short), 1, myFile);
				printf("Number of vertices: %d\n", quantity);
				
				for (int i = 0; i < 3*quantity; i++)
				{
					fread(&tempFloat, sizeof(float), 1, myFile);
					vertices.push_back(tempFloat);
				}
				break;
			case ID_FACES_DESCRIPTION:

				fread(&quantityPolygons, sizeof(unsigned short), 1, myFile);
				printf("Number of polygons: %d\n", quantityPolygons);

				for (int i = 0; i < quantityPolygons; i++)
				{
					fread(&tempIndex, sizeof(unsigned short), 1, myFile);
					indices.push_back(tempIndex);
					fread(&tempIndex, sizeof(unsigned short), 1, myFile);
					indices.push_back(tempIndex);
					fread(&tempIndex, sizeof(unsigned short), 1, myFile);
					indices.push_back(tempIndex);
					fread(&tempIndex, sizeof(unsigned short), 1, myFile);
				}
				break;
			case ID_MAPPING_COORDINATES_LIST:

				fread(&quantity, sizeof(unsigned short), 1, myFile);
				for (int i = 0; i < 2 * quantity; i++)
				{
					fread(&tempFloat, sizeof(float), 1, myFile);
					uvs.push_back(tempFloat);
				}
				break;
			default:
				//fseek(myFile, chunkLength - 6, SEEK_CUR);
				break;
		}
	}

	data->vertexCount = quantity;
	data->indexCount = 3* quantityPolygons;
	data->indexBuffer = new unsigned int[3 * quantityPolygons];
	data->vertexPositionBuffer = new GLfloat[3 * quantity];
	data->vertexColorBuffer = new GLfloat[3 * quantity];
	data->vertexUVBuffer = new GLfloat[2 * quantity];
	data->vertexNormalBuffer = new GLfloat[3 * quantity];

	list<GLfloat>::iterator itP, itUV;
	list<unsigned int>::iterator itI = indices.begin();
	itP = vertices.begin();
	itUV = uvs.begin();
	for (int i = 0; i < (3 * quantity); ++i, ++itP)
	{
		data->vertexPositionBuffer[i] = (*itP);
		data->vertexColorBuffer[i] = 1.0f;
	}

	for (int i = 0; i < (2 * quantity); ++i, ++itUV)
	{
		data->vertexUVBuffer[i] = (*itUV);
	}

	for (int i = 0; i < (3 * quantityPolygons); i++, ++itI)
	{
		data->indexBuffer[i] = (*itI);
	}

	glm::mat4 chuj = glm::rotate((3.14f / 2.0f), glm::vec3(1.0f, 0.0f, 0.0f));
	for (int i = 0; i < 3 * quantity; i += 3)
	{
		glm::vec3 dupa = glm::vec3(0.0f, 0.0f, -1.0f);
		dupa = glm::vec3(chuj * glm::vec4(dupa, 1.0f));
		data->vertexNormalBuffer[i] = dupa.x;
		data->vertexNormalBuffer[i + 1] = dupa.y;
		data->vertexNormalBuffer[i + 2] = dupa.z;
		printf((to_string(dupa.x) + " " + to_string(dupa.y)+ " " + to_string(dupa.z) + "\n").c_str());
	}
	//temporarily generate normals

	
	//glm::vec3 v1, v2, v3, edge1, edge2, normal;
	//for (int i = 0; i < (3 * quantity) - 6; i += 9)
	//{
	//	v1 = glm::vec3(data->vertexPositionBuffer[3*data->indexBuffer[i]], data->vertexPositionBuffer[3*data->indexBuffer[i]+1], data->vertexPositionBuffer[3*data->indexBuffer[i]+2]);
	//	v2 = glm::vec3(data->vertexPositionBuffer[3*data->indexBuffer[i + 1]], data->vertexPositionBuffer[3*data->indexBuffer[i + 1]+1], data->vertexPositionBuffer[3*data->indexBuffer[i + 1]+2]);
	//	v3 = glm::vec3(data->vertexPositionBuffer[3*data->indexBuffer[i + 2]], data->vertexPositionBuffer[3*data->indexBuffer[i + 2]+1], data->vertexPositionBuffer[3*data->indexBuffer[i + 2]+2]);
	//	edge1 = v2 - v1;
	//	edge2 = v3 - v1;
	//	normal = normalize(glm::cross(edge1, edge2));

	//	data->vertexNormalBuffer[data->indexBuffer[i]] = normal.x;
	//	data->vertexNormalBuffer[data->indexBuffer[i + 1]] = normal.y;
	//	data->vertexNormalBuffer[data->indexBuffer[i + 2]] = normal.z;
	//	data->vertexNormalBuffer[data->indexBuffer[i + 3]] = normal.x;
	//	data->vertexNormalBuffer[data->indexBuffer[i + 4]] = normal.y;
	//	data->vertexNormalBuffer[data->indexBuffer[i + 5]] = normal.z;
	//	data->vertexNormalBuffer[data->indexBuffer[i + 6]] = normal.x;
	//	data->vertexNormalBuffer[data->indexBuffer[i + 7]] = normal.y;
	//	data->vertexNormalBuffer[data->indexBuffer[i + 8]] = normal.z;
	//}
	///////

	Mesh* mesh = new Mesh();
	mesh->Initialize(m_programID, NULL, name, data);
	mesh->SetTexture(textures.at(0));
	AddMesh(mesh);

	oneByte = 5;
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