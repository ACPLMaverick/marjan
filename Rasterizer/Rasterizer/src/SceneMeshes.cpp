#include "SceneMeshes.h"
#include "Camera.h"
#include "Triangle.h"
#include "SpecificObjectFactory.h"
#include "System.h"
#include "Mesh.h"
#include "light/LightAmbient.h"
#include "light/LightDirectional.h"
#include "light/LightSpot.h"


SceneMeshes::SceneMeshes()
{
}


SceneMeshes::~SceneMeshes()
{
}

void SceneMeshes::InitializeScene()
{
	_cameras.push_back(new Camera(
		&math::Float3(3.0f, 3.0f, -10.0f),
		&math::Float3(0.0f, 0.0f, 0.0f),
		&math::Float3(0.0f, 1.0f, 0.0f),
		50.0f,
		(float)System::GetInstance()->GetSystemSettings()->GetDisplayWidth() /
		(float)System::GetInstance()->GetSystemSettings()->GetDisplayHeight(),
		1.0f,
		1.1f
	));

	std::string cPath = "canteen_albedo_specular";
	Texture* mat1Diff = new Texture(&cPath);
	cPath = "canteen_normals";
	Texture* mat1Nrm = new Texture(&cPath);
	Material* mat1 = new Material
	(
		mat1Diff,
		mat1Nrm,
		Color32(0xFFFFFFFF),
		Color32(0xFFFFFFFF),
		Color32(0xFFFFEEEE),
		60.0f
	);
	_materials.push_back(mat1);

	cPath = "janusz";
	Texture* mat2Diff = new Texture(&cPath);
	Texture* mat2Nrm = new Texture(Color32((uint8_t)255, 127, 127, 255));
	Material* mat2 = new Material
	(
		mat2Diff,
		mat2Nrm,
		Color32(0xFF00FF00),
		Color32(0xFF00FF00),
		Color32(0xFF00FFFF),
		5.0f
	);
	_materials.push_back(mat2);

	math::Float3 cPos(0.0f, 0.0f, 0.0f);
	math::Float3 cRot(0.0f, 0.0f, 0.0f);
	math::Float3 cScl(2.0f, 2.0f, 2.0f);
	cPath = "sphere";
	Mesh* m1 = SpecificObjectFactory::GetMesh(&cPos, &cRot, &cScl, &cPath);
	m1->SetMaterialPtr(mat1);
	_primitives.push_back(m1);

	cPath = "cube";
	cPos = math::Float3(-8.0f, 0.0f, 4.0f);
	Mesh* m2 = SpecificObjectFactory::GetMesh(&cPos, &cRot, &cScl, &cPath);
	m2->SetMaterialPtr(mat2);
	_primitives.push_back(m2);

	Color32 ambCol(1.0f, 0.15f, 0.1f, 0.2f);
	_lightAmbient = SpecificObjectFactory::GetLightAmbient(&ambCol);

	Color32 dirCol(1.0f, 1.0f, 1.0f, 1.0f);
	math::Float3 dirDir(-1.0f, -1.0f, 1.0f);
	_lightsDirectional.push_back(SpecificObjectFactory::GetLightDirectional(&dirCol, &dirDir));
}