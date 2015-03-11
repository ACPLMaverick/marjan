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

	int error = (fopen_s(&myFile, (*filePath).c_str(), "rb"));
	if (error != 0)
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

	// powrzucaæ vertexy do struktur vec3
	glm::vec3* ver = new glm::vec3[quantity];
	glm::vec3** verS = new glm::vec3*[quantityPolygons];
	for (int i = 0; i < quantityPolygons; i++)
		verS[i] = new glm::vec3[3];
	glm::vec3* nrmS = new glm::vec3[quantityPolygons];
	glm::vec3* nrm = new glm::vec3[quantity];
	for (int i = 0, j = 0; i < 3 * quantity; i+=3, ++j)
	{
		ver[j] = glm::vec3(data->vertexPositionBuffer[i], data->vertexPositionBuffer[i + 1], data->vertexPositionBuffer[i + 2]);
		nrm[j] = glm::vec3(0.0f, 0.0f, 0.0f);
	}
	// posortowaæ wg polygonów
	for (int i = 0, j = 0; i < 3 * quantityPolygons; i+=3, ++j)
	{
		verS[j][0] = glm::vec3(ver[data->indexBuffer[i]]);
		verS[j][1] = glm::vec3(ver[data->indexBuffer[i + 1]]);
		verS[j][2] = glm::vec3(ver[data->indexBuffer[i + 2]]);
	}
	// policzyæ normalne dla ka¿dego vertexa
	glm::vec3 e1, e2;
	for (int i = 0; i < quantityPolygons; i++)
	{
		e1 = verS[i][1] - verS[i][0];
		e2 = verS[i][2] - verS[i][0];
		nrmS[i] = glm::normalize(glm::cross(e1, e2));
	}
	// wgraæ je do wektora o d³ugoœci wierzcho³ków, normalizuj¹c
	for (int i = 0, j = -1; i < 3 * quantityPolygons; ++i)
	{
		if (i % 3 == 0) ++j;
		nrm[data->indexBuffer[i]] = glm::normalize(nrm[data->indexBuffer[i]] + nrmS[j]);
		
	}
	// sp³aszczyæ wektor normalnych bo te¿ ponoæ s¹ indeksowane
	for (int i = 0, j = 0; i < 3 * quantity; i+=3, ++j)
	{
		data->vertexNormalBuffer[i] = nrm[j].x;
		data->vertexNormalBuffer[i + 1] = nrm[j].y;
		data->vertexNormalBuffer[i + 2] = nrm[j].z;
		printf((to_string(nrm[j].x) + " " + to_string(nrm[j].y) + " " + to_string(nrm[j].z) + "\n").c_str());
	}

	delete[] ver;
	for (int i = 0; i < quantityPolygons; i++)
		delete[] verS[i];
	delete[] verS;
	delete[] nrmS;
	delete[] nrm;

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