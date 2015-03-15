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

	if(!Load3DS(&FILEPATH_FIXED)) return false;

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
	printf("Loading 3DS \"%s\" ... \n", (*filePath).c_str());
	string name;
	string textureName;
	FILE* myFile;
	list<GLfloat> vertices, uvs;
	list<unsigned int> indices;
	unsigned short chunkID;
	GLuint chunkLength;
	unsigned char oneByte;
	unsigned short quantity = 0;
	unsigned short quantityPolygons = 0;
	unsigned short faceFlags;
	GLfloat localTransform[4][4];

	vector<Mesh*> hierarchyArray;
	Mesh* hPtr;
	short myID = -2, parentID = -2;
	unsigned int counter = -1;

	char tempChar = 'a';
	GLfloat tempFloat;
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
				// this is the beginning of loading new mesh. Clearing the lists and values then.
				counter++;
				name = "";
				vertices.clear();
				uvs.clear();
				indices.clear();
				quantity = 0;
				quantityPolygons = 0;
				tempChar = 'a';	// <-- this makes you unable to load more meshes than one
				////////////////////////////
				for (int i = 0; i < 20 && tempChar != '\0'; i++)
				{
					fread(&tempChar, 1, 1, myFile);
					name.push_back(tempChar);
				}
				printf(("Name: " + name).c_str());
				printf(("\n"));
				break;
			case ID_TRIANGULAR_MESH:
				oneByte = 5;
				break;
			case ID_TRIANGULAR_MESH_02:
				oneByte = 5;
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

				// this is the end of loading a single mesh
				hPtr = GenerateMesh
					(
						&name,
						&textureName,
						&vertices,
						&uvs,
						&indices,
						quantity,
						quantityPolygons,
						localTransform,
						myID,
						parentID
					);
				hierarchyArray.push_back(hPtr);
				//////////////////
				break;
			case ID_LOCAL_COORDINATES_SYSTEM:
				//fseek(myFile, chunkLength - 6, SEEK_CUR);
				for (int i = 0; i < 4; i++)
				{
					for (int j = 0; j < 3; j++)
					{
						fread(&tempFloat, sizeof(float), 1, myFile);
						localTransform[i][j] = tempFloat;
					}
					localTransform[i][3] = 1.0f;
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
			case ID_MATERIAL_BLOCK:
				oneByte = 5;
				break;
			case ID_DIFFUSE_COLOR:
				// tu powinna byæ nazwa tekstury
				for (int i = 0; i < 26; i++)
				{
					fread(&tempChar, 1, 1, myFile);
					if(i > 13 && i < 26) textureName.push_back(tempChar);
				}
				printf(("Texture name: " + textureName + "\n").c_str());
				myFile->_ptr -= 26;
				myFile->_cnt += 26;
				oneByte = 5;
				break;
			default:
				fseek(myFile, chunkLength - 6, SEEK_CUR);
				break;
		}
	}

	// reading file once more for hierarchy positions and in result number of meshes;
	// B010 - name and hierarchy parent B030 - hierarchy number
	rewind(myFile);
	signed short hier = 0;
	signed short father = 0;
	counter = -1;
	while (ftell(myFile) < _filelength(_fileno(myFile)))
	{
		fread(&chunkID, 2, 1, myFile);
		fread(&chunkLength, 4, 1, myFile);

		switch (chunkID)
		{
		case ID_MAIN_CHUNK:
			break;
		case ID_EDITOR_CHUNK:
			break;
		case ID_KEYFRAMER_CHUNK:
			break;
		case ID_FRAMES:
			fseek(myFile, chunkLength - 6, SEEK_CUR);
			break;
		case ID_UNKNOWN_01:
			counter++;
			break;
		case ID_HIERARCHY_POSITION:
			hier = 0;
			fread(&hier, 2, 1, myFile);
			break;
		case ID_OBJECT_NAME:
			fseek(myFile, chunkLength - 8, SEEK_CUR);
			fread(&father, 2, 1, myFile);

			// saving to hierarchy vector
			hierarchyArray[counter]->myID = hier;
			hierarchyArray[counter]->parentID = father;
			break;
		default:
			fseek(myFile, chunkLength - 6, SEEK_CUR);
			break;
		}
	}

	printf("Finished loading \"%s\". Generating models...\n", (*filePath).c_str());

	// sort out hierarchical shit
	if(hierarchyArray.size() > 1) SolveHierarchy(&hierarchyArray);
	else if (hierarchyArray.size() == 1) AddMesh(hierarchyArray[0]);
	else
	{
		printf("3DS loading failed - no meshes found.\n");
		return false;
	}

	//for (vector<MeshHierarchy*>::iterator it = hierarchyArray.begin(); it != hierarchyArray.end(); ++it)
	//{
	//	AddMesh((*it)->mesh);
	//	delete (*it);
	//}

	printf("3DS Successfully loaded.\n");
	return true;
}

// this function is mainly for generating normals and filling mesh with data, and then - initializing it
Mesh* MeshManager::GenerateMesh(string* name, string* textureName,
	list<GLfloat>* vertices, list<GLfloat>* uvs, list<unsigned int>* indices,
	unsigned short quantity, unsigned short quantityPolygons, GLfloat localTrans[][4],
	short myID, short parentID)
{
	printf("Generating mesh %s \n", name->c_str());

	VertexData* data = new VertexData;

	data->vertexCount = quantity;
	data->indexCount = 3 * quantityPolygons;
	data->indexBuffer = new unsigned int[3 * quantityPolygons];
	data->vertexPositionBuffer = new GLfloat[3 * quantity];
	data->vertexColorBuffer = new GLfloat[3 * quantity];
	data->vertexUVBuffer = new GLfloat[2 * quantity];
	data->vertexNormalBuffer = new GLfloat[3 * quantity];

	list<GLfloat>::iterator itP, itUV;
	list<unsigned int>::iterator itI = indices->begin();
	itP = vertices->begin();
	itUV = uvs->begin();
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
	for (int i = 0, j = 0; i < 3 * quantity; i += 3, ++j)
	{
		ver[j] = glm::vec3(data->vertexPositionBuffer[i], data->vertexPositionBuffer[i + 1], data->vertexPositionBuffer[i + 2]);
		nrm[j] = glm::vec3(0.0f, 0.0f, 0.0f);
	}
	// posortowaæ wg polygonów
	for (int i = 0, j = 0; i < 3 * quantityPolygons; i += 3, ++j)
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
	for (int i = 0, j = 0; i < 3 * quantity; i += 3, ++j)
	{
		data->vertexNormalBuffer[i] = nrm[j].x;
		data->vertexNormalBuffer[i + 1] = nrm[j].y;
		data->vertexNormalBuffer[i + 2] = nrm[j].z;
	}

	delete[] ver;
	for (int i = 0; i < quantityPolygons; i++)
		delete[] verS[i];
	delete[] verS;
	delete[] nrmS;
	delete[] nrm;

	// solving transformations...
	glm::vec3 pivotPos = glm::vec3(localTrans[3][0], localTrans[3][1], localTrans[3][2]);
	glm::mat4 mat = glm::translate(pivotPos);
	for (int i = 0; i < 3 * data->vertexCount; i += 3)
	{
		data->vertexPositionBuffer[i] -= pivotPos.x;
		data->vertexPositionBuffer[i + 1] -= pivotPos.y;
		data->vertexPositionBuffer[i + 2] -= pivotPos.z;
	}

	// calculating bounding sphere
	GLfloat radius = 0.0f;
	GLfloat tempDist = 0.0f;
	glm::vec3 maxPosA = glm::vec3(0.0f, 0.0f, 0.0f);
	glm::vec3 maxPosB = glm::vec3(0.0f, 0.0f, 0.0f);
	glm::vec3 tempPosA = glm::vec3(0.0f, 0.0f, 0.0f);
	glm::vec3 tempPosB = glm::vec3(0.0f, 0.0f, 0.0f);
	glm::vec3 centerPos = glm::vec3(0.0f, 0.0f, 0.0f);
	for (int i = 0; i < 3 * data->vertexCount; i += 3)
	{
		tempPosA.x = data->vertexPositionBuffer[i];
		tempPosA.y = data->vertexPositionBuffer[i + 1];
		tempPosA.z = data->vertexPositionBuffer[i + 2];
		for (int j = 0; j < 3 * data->vertexCount; j += 3)
		{
			tempPosB.x = data->vertexPositionBuffer[j];
			tempPosB.y = data->vertexPositionBuffer[j + 1];
			tempPosB.z = data->vertexPositionBuffer[j + 2];
			tempDist = glm::length(tempPosA - tempPosB);
			if (tempDist > radius)
			{
				radius = tempDist;
				maxPosA = tempPosA;
				maxPosB = tempPosB;
			}
		}
	}
	radius /= 2.0f;
	centerPos.x = (maxPosA.x + maxPosB.x) / 2.0f;
	centerPos.y = (maxPosA.y + maxPosB.y) / 2.0f;
	centerPos.z = (maxPosA.z + maxPosB.z) / 2.0f;

	BoundingSphere* sphere = new BoundingSphere;
	sphere->d_position = glm::vec4(centerPos, 1.0f);
	sphere->d_radius = radius;
	sphere->position = glm::vec4(centerPos, 1.0f);
	sphere->radius = radius;

	Mesh* mesh = new Mesh();
	mesh->Initialize(m_programID, NULL, *name, data, sphere, myID, parentID);

	// loadin textures
	if (textureName->size() > 0)
	{
		Texture* m_texture = new Texture();
		if (!m_texture->Initialize(textureName)) return false;
		mesh->SetTexture(m_texture);
	}

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

	// returning to origin
	//mesh->Transform(&(pivotPos), &glm::vec3(0.0f, 0.0f, 0.0f), &glm::vec3(1.0f, 1.0f, 1.0f));

	printf("Mesh %s successfully generated.\n", name->c_str());
	return mesh;
}

void MeshManager::SolveHierarchy(vector<Mesh*>* meshHierarchies)
{
	unsigned int currentLevel = 0;
	Mesh* lastPtr = (*meshHierarchies)[0];
	if (lastPtr->parentID == -1) AddMesh(lastPtr);
	Mesh* currentPtr;
	for (vector<Mesh*>::iterator it = meshHierarchies->begin() + 1; it != meshHierarchies->end(); ++it)
	{
		currentPtr = (*it);
		if (currentPtr->parentID == -1)
		{
			AddMesh(currentPtr);
			lastPtr = currentPtr;
		}
		if (lastPtr->myID == currentPtr->parentID)
		{
			lastPtr->AddChild(currentPtr);
			lastPtr = currentPtr;
		}
		if (lastPtr->myID < currentPtr->parentID)
		{
			lastPtr = lastPtr->GetParent();
			--it;
		}
	}
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

vector<Mesh*>* MeshManager::GetMeshCollection()
{
	return &meshes;
}