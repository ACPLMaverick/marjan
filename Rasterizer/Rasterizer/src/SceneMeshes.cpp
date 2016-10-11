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

	math::Float3 cPos(0.0f, 0.0f, 0.0f);
	math::Float3 cRot(0.0f, 0.0f, 0.0f);
	math::Float3 cScl(2.0f, 2.0f, 2.0f);
	std::string cPath = "sphere";
	_primitives.push_back(SpecificObjectFactory::GetMesh(&cPos, &cRot, &cScl, &cPath));

	cPath = "cube";
	cPos = math::Float3(-8.0f, 0.0f, 4.0f);
	_primitives.push_back(SpecificObjectFactory::GetMesh(&cPos, &cRot, &cScl, &cPath));

	Color32 ambCol(1.0f, 0.15f, 0.1f, 0.2f);
	_lightAmbient = SpecificObjectFactory::GetLightAmbient(&ambCol);

	Color32 dirCol(1.0f, 1.0f, 1.0f, 1.0f);
	math::Float3 dirDir(-1.0f, -1.0f, 1.0f);
	_lightsDirectional.push_back(SpecificObjectFactory::GetLightDirectional(&dirCol, &dirDir));
}