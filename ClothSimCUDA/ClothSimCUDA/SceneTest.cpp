#include "SceneTest.h"
#include "MeshGLRect.h"

SceneTest::SceneTest(string n) : Scene(n)
{
}

SceneTest::SceneTest(const SceneTest* c) : Scene(c)
{
}


SceneTest::~SceneTest()
{
}



unsigned int SceneTest::Initialize()
{
	SimObject* controller = new SimObject();
	controller->Initialize("SimController");

	SimController* sc = new SimController(controller);
	sc->Initialize();
	controller->AddComponent(sc);

	m_objects.push_back(controller);


	SimObject* testObj = new SimObject();
	testObj->Initialize("testObj");

	Transform* testObjTransform = new Transform(testObj);
	testObjTransform->Initialize();
	testObjTransform->SetPosition(&(glm::vec3(0.0f, 0.0f, 0.0f)));
	testObjTransform->SetScale(&(glm::vec3(3.0f, 1.0f, 1.0f)));
	testObj->SetTransform(testObjTransform);

	MeshGLRect* triangle = new MeshGLRect(testObj);
	triangle->Initialize();
	testObj->AddMesh(triangle);

	RotateMe* rm = new RotateMe(testObj);
	rm->Initialize();
	rm->SetRotation(&(glm::vec3(0.0f, 0.00003f, 0.0f)));
	testObj->AddComponent(rm);

	m_objects.push_back(testObj);


	Camera* testCam = new Camera(nullptr, 0.6f, 0.01f, 1000.0f);
	testCam->Initialize();
	testCam->SetPosition(&glm::vec3(-8.0f, 15.0f, 15.0f));

	m_cameras.push_back(testCam);


	m_currentCameraID = 0;
	m_currentObjectID = 0;


	SetAmbientLight(new LightAmbient(&(glm::vec3(0.1f, 0.1f, 0.2f))));
	LightDirectional* dir1 = new LightDirectional(&(glm::vec3(1.0f, 0.8f, 0.8f)), &(glm::vec3(0.0f, 1.0f, 0.0f)), &(glm::vec3(-1.0f, -1.0f, -1.0f)));
	AddDirectionalLight(dir1);

#ifdef _DEBUG
	printf("\SceneTest.Initialize has completed successfully.\n");
#endif

	return CS_ERR_NONE;
}