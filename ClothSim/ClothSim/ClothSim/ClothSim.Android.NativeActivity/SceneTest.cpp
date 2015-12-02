#include "SceneTest.h"
#include "MeshGLRect.h"
#include "MeshGLBox.h"
#include "MeshGLPlane.h"
#include "MeshGLSphere.h"
#include "GUIText.h"
#include "BoxAACollider.h"
#include "SphereCollider.h"
#include "ClothSimulator.h"
#include "GUIButton.h"

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

	glm::vec3 tPos, tRot, tScl, tPosMin, tPosMax;
	glm::vec4 tCol;


	SimObject* ground = new SimObject();
	ground->Initialize("Ground");

	tPos = (glm::vec3(0.0f, -10.0f, 0.0f));
	tRot = (glm::vec3(-3.14f / 2.0f, 0.0f, 0.0f));
	tScl = (glm::vec3(100.0f, 100.0f, 100.0f));
	tCol = (glm::vec4(0.6f, 0.6f, 0.6f, 1.0f));

	Transform* groundTransform = new Transform(ground);
	groundTransform->Initialize();
	groundTransform->SetPosition(&tPos);
	groundTransform->SetRotation(&tRot);
	groundTransform->SetScale(&tScl);
	ground->SetTransform(groundTransform);
	MeshGLRect* triangle = new MeshGLRect(ground, &tCol);
	triangle->Initialize();
	//triangle->SetGloss(100.0f);
	triangle->SetTextureID(ResourceManager::GetInstance()->GetTextureWhite());
	ground->AddMesh(triangle);

	AddObject(ground);


	SimObject* testObj = new SimObject();
	testObj->Initialize("testObj");

	tPos = (glm::vec3(0.0f, 2.5f, 0.0f));
	tScl = (glm::vec3(1.0f, 1.0f, 1.0f));
	tCol = (glm::vec4(0.4f, 0.7f, 0.9f, 1.0f));

	Transform* testObjTransform = new Transform(testObj);
	testObjTransform->Initialize();
	testObjTransform->SetPosition(&tPos);
	testObjTransform->SetScale(&tScl);
	testObj->SetTransform(testObjTransform);
	/*
	MeshGLBox* box = new MeshGLBox(testObj, 4.0f, 3.0f, 3.0f, &(glm::vec4(0.2f, 0.2f, 0.8f, 1.0f)));
	box->Initialize();
	box->SetGloss(100.0f);
	box->SetTextureID(ResourceManager::GetInstance()->GetTextureWhite());
	testObj->AddMesh(box);
	*/
	
	MeshGLSphere* sph = new MeshGLSphere(testObj, 2.0f, 32, 32, &tCol);
	sph->Initialize();
	sph->SetGloss(20.0f);
	sph->SetTextureID(ResourceManager::GetInstance()->GetTextureWhite());
	testObj->AddMesh(sph);

	/*
	BoxAACollider* tObjCollider = PhysicsManager::GetInstance()->CreateBoxAACollider(testObj, &(glm::vec3(-2.0f, -1.5f, -1.5f)), &(glm::vec3(2.0f, 1.5f, 1.5f)));
	testObj->AddCollider(tObjCollider);
	*/

	tPosMin = (glm::vec3(0.0f, 0.0f, 0.0f));

	PhysicsManager* ph = PhysicsManager::GetInstance();

	SphereCollider* tObjCollider = PhysicsManager::GetInstance()->CreateSphereCollider(testObj, &tPosMin, 2.0f);
	testObj->AddCollider(tObjCollider);

	

	AddObject(testObj);

	m_currentObjectID = 1;


	SimObject* colObj = new SimObject();
	colObj->Initialize("colObj");

	tPos = (glm::vec3(5.0f, 2.5f, 0.0f));
	tScl = (glm::vec3(1.0f, 1.0f, 1.0f));
	tCol = (glm::vec4(0.8f, 0.2f, 0.2f, 1.0f));
	tPosMin = (glm::vec3(-0.5f, -0.5f, -0.5f));
	tPosMax = (glm::vec3(0.5f, 0.5f, 0.5f));

	Transform* colObjTransform = new Transform(colObj);
	colObjTransform->Initialize();
	colObjTransform->SetPosition(&tPos);
	colObjTransform->SetScale(&tScl);
	colObj->SetTransform(colObjTransform);
	MeshGLBox* colBox = new MeshGLBox(colObj, 1.0f, 1.0f, 1.0f, &tCol);
	colBox->Initialize();
	colBox->SetGloss(60.0f);
	colBox->SetTextureID(ResourceManager::GetInstance()->GetTextureWhite());
	colObj->AddMesh(colBox);
	BoxAACollider* cObjCollider = PhysicsManager::GetInstance()->CreateBoxAACollider(colObj, &tPosMin, &tPosMax);
	colObj->AddCollider(cObjCollider);

	RotateMe* rm = new RotateMe(colObj);
	rm->Initialize();
	tRot = glm::vec3(0.0f, 0.003f, 0.0f);
	rm->SetRotation(&tRot);
	colObj->AddComponent(rm);

	AddObject(colObj);


	SimObject* testCloth = new SimObject();
	testCloth->Initialize("testCloth");

	tPos = (glm::vec3(0.0f, 7.5f, 0.0f));
	tScl = (glm::vec3(1.0f, 1.0f, 1.0f));
	tCol = glm::vec4(1.0f, 0.5f, 0.7f, 1.0f);

	Transform* testClothTransform = new Transform(testCloth);
	testClothTransform->Initialize();
	testClothTransform->SetPosition(&tPos);
	testClothTransform->SetScale(&tScl);
	testClothTransform->Update();
	testCloth->SetTransform(testClothTransform);

	MeshGLPlane* clothMesh = new MeshGLPlane(testCloth, 10.0f, 10.0f, 20, 20, &tCol);
	clothMesh->Initialize();
	clothMesh->SetTextureID(ResourceManager::GetInstance()->GetTextureWhite());
	testCloth->AddMesh(clothMesh);
	/*
	ClothSimulator* cSim = new ClothSimulator(testCloth, 1);
	testCloth->AddComponent(cSim);
	cSim->Initialize();
	*/
	AddObject(testCloth);

	////////////////////////
	/////////// Camera

	Camera* testCam = new Camera(nullptr, 0.6f, 0.01f, 1000.0f);
	testCam->Initialize();
	
	tPos = glm::vec3(-8.0f, 15.0f, 15.0f) * 2.0f;
	testCam->SetPosition(&tPos);

	AddCamera(testCam);


	m_currentCameraID = 0;
	

	////////////////////////
	/////////// Lights

	glm::vec3 aCol = (glm::vec3(0.1f, 0.05f, 0.1f));
	glm::vec3 ldCol = (glm::vec3(1.0f, 0.9f, 0.6f));
	glm::vec3 ldSpec = (glm::vec3(1.0f, 0.9f, 0.9f));
	glm::vec3 ldDir = (glm::vec3(-0.8f, -0.8f, -1.0f));
	LightAmbient* la = new LightAmbient(&aCol);
	SetAmbientLight(la);
	LightDirectional* dir1 = new LightDirectional(&ldCol, &ldSpec, &ldDir);
	AddDirectionalLight(dir1);

	////////////////////////
	/////////// GUI

	string t1n = "FPStitle";
	string t1v = "FPS: ";
	string t2n = "DeltaTimetitle";
	string t2v = "Delta time [ms]: ";
	string t3n = "TotalTimetitle";
	string t3v = "Total time [s]: ";
	string tb1 = "BtnExit";
	string tval01 = "FPSvalue";
	string tval02 = "DTvalue";
	string tval03 = "TTvalue";
	string dummy = "Dummy";
	string tex = "textures/ExportedFont.bmp";
	glm::vec2 scl = glm::vec2(0.025f, 0.025f);

	GUIText* gt = new GUIText(&t1n, &t1v, ResourceManager::GetInstance()->GetTexture(&tex));
	gt->Initialize();
	gt->SetPosition(glm::vec2(-0.95f, 0.85f));
	gt->SetScale(scl);

	GUIText* gt2 = new GUIText(&t2n, &t2v, ResourceManager::GetInstance()->GetTexture(&tex));
	gt2->Initialize();
	gt2->SetPosition(glm::vec2(-0.95f, 0.78f));
	gt2->SetScale(scl);

	GUIText* gt3 = new GUIText(&t3n, &t3v, ResourceManager::GetInstance()->GetTexture(&tex));
	gt3->Initialize();
	gt3->SetPosition(glm::vec2(-0.95f, 0.71f));
	gt3->SetScale(scl);

	GUIText* gt4 = new GUIText(&tval01, &dummy, ResourceManager::GetInstance()->GetTexture(&tex));
	gt4->Initialize();
	gt4->SetPosition(glm::vec2(-0.22f, 0.85f));
	gt4->SetScale(scl);

	GUIText* gt5 = new GUIText(&tval02, &dummy, ResourceManager::GetInstance()->GetTexture(&tex));
	gt5->Initialize();
	gt5->SetPosition(glm::vec2(-0.22f, 0.78f));
	gt5->SetScale(scl);

	GUIText* gt6 = new GUIText(&tval03, &dummy, ResourceManager::GetInstance()->GetTexture(&tex));
	gt6->Initialize();
	gt6->SetPosition(glm::vec2(-0.22f, 0.71f));
	gt6->SetScale(scl);

	GUIButton* gb1 = new GUIButton(&tb1);
	gb1->Initialize();
	gb1->SetTextures(ResourceManager::GetInstance()->GetTexture(&tex), ResourceManager::GetInstance()->GetTexture(&tex));
	gb1->SetPosition(glm::vec2(0.0f, 0.0f));
	gb1->SetScale(glm::vec2(0.3f, 0.3f));

	AddGUIElement(gt);
	AddGUIElement(gt2);
	AddGUIElement(gt3);
	AddGUIElement(gt4);
	AddGUIElement(gt5);
	AddGUIElement(gt6);
	AddGUIElement(gb1);

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