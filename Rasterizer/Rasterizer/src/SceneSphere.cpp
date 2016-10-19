#include "SceneSphere.h"
#include "Camera.h"
#include "SpecificObjectFactory.h"
#include "System.h"
#include "Sphere.h"


SceneSphere::SceneSphere()
{
}


SceneSphere::~SceneSphere()
{
}

void SceneSphere::InitializeScene()
{
	_cameras.push_back(new Camera(
		&math::Float3(-2.0f, -1.5f, -5.0f),
		&math::Float3(0.0f, 0.0f, 0.0f),
		&math::Float3(0.0f, 1.0f, 0.0f),
		50.0f,
		(float)System::GetInstance()->GetSystemSettings()->GetDisplayWidth() /
		(float)System::GetInstance()->GetSystemSettings()->GetDisplayHeight()
		));

	Sphere* sph = new Sphere(math::Float3(0.0f, 0.0f, 0.0f), 1.0f);
	Material* mat = new Material();
	sph->SetMaterialPtr(mat);
	_materials.push_back(mat);
	_primitives.push_back(sph);

	Sphere* sph2 = new Sphere(math::Float3(2.0f, 0.0f, 4.0f), 1.0f);
	Material* mat2 = new Material(nullptr, nullptr, Color32(0x00000000), Color32(0xFFFF0000));
	sph2->SetMaterialPtr(mat2);
	_materials.push_back(mat2);
	_primitives.push_back(sph2);
}
