#include "SceneTest.h"
#include "MeshGLRect.h"
#include "GUIText.h"

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
	////////////////////////
	/////////// Objects

	SimObject* testObj = new SimObject();
	testObj->Initialize("testObj");

	Transform* testObjTransform = new Transform(testObj);
	testObjTransform->Initialize();
	testObjTransform->SetPosition(&(glm::vec3(0.0f, 0.0f, 0.0f)));
	testObjTransform->SetScale(&(glm::vec3(3.0f, 1.0f, 1.0f)));
	testObj->SetTransform(testObjTransform);

	MeshGLRect* triangle = new MeshGLRect(testObj);
	triangle->Initialize();
	triangle->SetTextureID(ResourceManager::GetInstance()->GetTextureWhite());
	testObj->AddMesh(triangle);

	/*RotateMe* rm = new RotateMe(testObj);
	rm->Initialize();
	rm->SetRotation(&(glm::vec3(0.0f, 0.00003f, 0.0f)));
	testObj->AddComponent(rm);*/

	AddObject(testObj);

	m_currentObjectID = 0;

	////////////////////////
	/////////// Camera

	Camera* testCam = new Camera(nullptr, 0.6f, 0.01f, 1000.0f);
	testCam->Initialize();
	testCam->SetPosition(&glm::vec3(-8.0f, 15.0f, 15.0f));

	AddCamera(testCam);


	m_currentCameraID = 0;
	

	////////////////////////
	/////////// Lights

	SetAmbientLight(new LightAmbient(&(glm::vec3(0.1f, 0.1f, 0.2f))));
	LightDirectional* dir1 = new LightDirectional(&(glm::vec3(1.0f, 0.8f, 0.8f)), &(glm::vec3(0.0f, 1.0f, 0.0f)), &(glm::vec3(-1.0f, -1.0f, -1.0f)));
	AddDirectionalLight(dir1);

	////////////////////////
	/////////// GUI

	string t1n = "FPStitle";
	string t1v = "FPS: ";
	string t2n = "DeltaTimetitle";
	string t2v = "Delta time [ms]: ";
	string t3n = "TotalTimetitle";
	string t3v = "Total time [ms]: ";
	string tval01 = "FPSvalue";
	string tval02 = "DTvalue";
	string tval03 = "TTvalue";
	string dummy = "Dummy";
	string tex = "..\\files\\ExportedFont.bmp";

	GUIText* gt = new GUIText(&t1n, &t1v, ResourceManager::GetInstance()->GetTexture(&tex));
	gt->Initialize();
	gt->SetPosition(glm::vec2(-0.9f, 0.8f));
	gt->SetScale(glm::vec2(0.02f, 0.02f));

	GUIText* gt2 = new GUIText(&t2n, &t2v, ResourceManager::GetInstance()->GetTexture(&tex));
	gt2->Initialize();
	gt2->SetPosition(glm::vec2(-0.9f, 0.7f));
	gt2->SetScale(glm::vec2(0.02f, 0.02f));

	GUIText* gt3 = new GUIText(&t3n, &t3v, ResourceManager::GetInstance()->GetTexture(&tex));
	gt3->Initialize();
	gt3->SetPosition(glm::vec2(-0.9f, 0.6f));
	gt3->SetScale(glm::vec2(0.02f, 0.02f));

	GUIText* gt4 = new GUIText(&tval01, &dummy, ResourceManager::GetInstance()->GetTexture(&tex));
	gt4->Initialize();
	gt4->SetPosition(glm::vec2(-0.79f, 0.8f));
	gt4->SetScale(glm::vec2(0.02f, 0.02f));

	GUIText* gt5 = new GUIText(&tval02, &dummy, ResourceManager::GetInstance()->GetTexture(&tex));
	gt5->Initialize();
	gt5->SetPosition(glm::vec2(-0.55f, 0.7f));
	gt5->SetScale(glm::vec2(0.02f, 0.02f));

	GUIText* gt6 = new GUIText(&tval03, &dummy, ResourceManager::GetInstance()->GetTexture(&tex));
	gt6->Initialize();
	gt6->SetPosition(glm::vec2(-0.55f, 0.6f));
	gt6->SetScale(glm::vec2(0.02f, 0.02f));

	AddGUIElement(gt);
	AddGUIElement(gt2);
	AddGUIElement(gt3);
	AddGUIElement(gt4);
	AddGUIElement(gt5);
	AddGUIElement(gt6);

	////////////////////////
	/////////// Controllers

	SimObject* controller = new SimObject();
	controller->Initialize("SimController");

	GUIController* sc = new GUIController(controller);
	sc->Initialize();
	controller->AddComponent(sc);

	AddObject(controller);

#ifdef _DEBUG
	printf("\SceneTest.Initialize has completed successfully.\n");
#endif

	return CS_ERR_NONE;
}