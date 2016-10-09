#include "SceneMeshes.h"
#include "Camera.h"
#include "Triangle.h"
#include "SpecificObjectFactory.h"
#include "System.h"
#include "Mesh.h"

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
			(float)System::GetInstance()->GetSystemSettings()->GetDisplayHeight()
	));

	math::Float3 cPos(0.0f, 0.0f, 0.0f);
	math::Float3 cRot(0.0f, 0.0f, 0.0f);
	math::Float3 cScl(1.0f, 1.0f, 1.0f);
	std::string cPath = "cube";
	_primitives.push_back(SpecificObjectFactory::GetMesh(&cPos, &cRot, &cScl, &cPath));
}