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
	SimObject* testObj = new SimObject();
	testObj->Initialize("testObj");
	Transform* testObjTransform = new Transform(testObj);
	MeshGLRect* triangle = new MeshGLRect(testObj);
	testObj->SetTransform(testObjTransform);
	testObj->AddMesh(triangle);
	testObjTransform->Initialize();
	triangle->Initialize();

	m_objects.push_back(testObj);


	Camera* testCam = new Camera(nullptr, 0.6f, 0.01f, 1000.0f);
	testCam->Initialize();

	m_cameras.push_back(testCam);


	m_currentCameraID = 0;
	m_currentObjectID = 0;

#ifdef _DEBUG
	printf("\SceneTest.Initialize has completed successfully.\n");
#endif

	return CS_ERR_NONE;
}