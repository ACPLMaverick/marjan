#pragma once
#include <GL\glew.h>
#include <GLFW\glfw3.h>
#include <glm\glm\glm.hpp>
#include <glm\glm\gtx\transform.hpp>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>

#include "Texture.h"
#include "Light.h"
#include "Mesh.h"

#define ID_MAIN_CHUNK 0x4D4D
#define ID_EDITOR_CHUNK 0x3D3D
#define ID_OBJECT_BLOCK 0x4000
#define ID_TRIANGULAR_MESH 0x4100
#define ID_VERTICES_LIST 0x4110
#define ID_FACES_DESCRIPTION 0x4120
#define ID_MAPPING_COORDINATES_LIST 0x4140
#define ID_SMOOTHING_GROUP_LIST 0x4150
#define ID_LOCAL_COORDINATES_SYSTEM 0x4160
#define ID_KEYFRAMER_CHUNK 0xB000
#define ID_FRAMES 0xB008
#define ID_OBJECT_NAME 0xB010
#define ID_HIERARCHY_POSITION 0xB030

static const string TEST_DIFFUSE = "E:\\_projects\\Cave\\canteen\\maps\\canteen_albedo.dds";
static const string TEST_SPECULAR = "E:\\_projects\\Engine2DAssets\\water02.dds";
static const string FILEPATH_FIXED = "E:\\_projects\\Cave\\canteen\\mesh\\Canteen.3DS";

using namespace std;

class MeshManager
{
private:
	vector<Mesh*> meshes;
	vector<Texture*> textures;
public:
	MeshManager();
	~MeshManager();

	bool Initialize(GLuint programID);
	void Shutdown();

	void Draw(glm::mat4* projectionMatrix, glm::mat4* viewMatrix, glm::vec3* eyeVector, GLuint eyeVectorID, Light* light);

	bool Load3DS(string filePath);

	void AddMesh(Mesh* mesh);
	Mesh* GetMesh(unsigned int i);
};

