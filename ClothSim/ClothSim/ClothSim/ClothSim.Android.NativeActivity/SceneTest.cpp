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
#include "GUIAction.h"
#include "GUIActionExitProgram.h"
#include "GUIActionSetDisplayMode.h"
#include "GUIActionShowPreferences.h"
#include "GUIActionMoveActiveObject.h"

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
	string tb2 = "BtnPreferences";
	string tb3 = "BtnWireframe";
	string tb4 = "BtnArrowFw";
	string tb5 = "BtnArrowBw";
	string tb6 = "BtnArrowLeft";
	string tb7 = "BtnArrowRight";
	string tb8 = "BtnArrowUp";
	string tb9 = "BtnArrowDown";
	string tval01 = "FPSvalue";
	string tval02 = "DTvalue";
	string tval03 = "TTvalue";
	string dummy = "Dummy";
	string tex = "textures/ExportedFont.tga";
	string tBtnEx = "textures/btn_exit.png";
	string tBtnExA = "textures/btn_exit_a.png";
	string tBtnWf = "textures/btn_wireframe.png";
	string tBtnWfA = "textures/btn_wireframe_a.png";
	string tBtnSt = "textures/btn_settings.png";
	string tBtnStA = "textures/btn_settings_a.png";
	string tBtnArr = "textures/btn_arrow_up.png";
	string tBtnArrA = "textures/btn_arrow_up_a.png";
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
	gb1->SetTextures(ResourceManager::GetInstance()->LoadTexture(&tBtnEx), ResourceManager::GetInstance()->LoadTexture(&tBtnExA));
	gb1->SetPosition(glm::vec2(0.7f, 0.8f));
	gb1->SetScale(glm::vec2(0.15f, 0.15f));
	GUIAction* gb1a = new GUIActionExitProgram(gb1);
	gb1->AddActionClick(gb1a);

	GUIButton* gb2 = new GUIButton(&tb2);
	gb2->Initialize();
	gb2->SetTextures(ResourceManager::GetInstance()->LoadTexture(&tBtnSt), ResourceManager::GetInstance()->LoadTexture(&tBtnStA));
	gb2->SetPosition(glm::vec2(0.25f, -0.75f));
	gb2->SetScale(glm::vec2(0.2f, 0.2f));
	GUIAction* gb2a = new GUIActionShowPreferences(gb2);
	gb2->AddActionClick(gb2a);

	GUIButton* gb3 = new GUIButton(&tb3);
	gb3->Initialize();
	gb3->SetTextures(ResourceManager::GetInstance()->LoadTexture(&tBtnWf), ResourceManager::GetInstance()->LoadTexture(&tBtnWfA));
	gb3->SetPosition(glm::vec2(0.7f, -0.75f));
	gb3->SetScale(glm::vec2(0.2f, 0.2f));
	GUIAction* gb3a = new GUIActionSetDisplayMode(gb3);
	gb3->AddActionClick(gb3a);

	GUIButton* gb4 = new GUIButton(&tb4);
	gb4->Initialize();
	gb4->SetTextures(ResourceManager::GetInstance()->LoadTexture(&tBtnArr), ResourceManager::GetInstance()->LoadTexture(&tBtnArrA));
	gb4->SetPosition(glm::vec2(-0.4f, -0.3f));
	gb4->SetRotation(0.0f);
	gb4->SetScale(glm::vec2(0.15f, 0.15f));
	gb4->SetParamsHold((void*)1);
	GUIAction* gb4a = new GUIActionMoveActiveObject(gb4);
	gb4->AddActionHold(gb4a);

	GUIButton* gb5 = new GUIButton(&tb5);
	gb5->Initialize();
	gb5->SetTextures(ResourceManager::GetInstance()->LoadTexture(&tBtnArr), ResourceManager::GetInstance()->LoadTexture(&tBtnArrA));
	gb5->SetPosition(glm::vec2(-0.4f, -0.8f));
	gb5->SetRotation(M_PI * 0.5f);
	gb5->SetScale(glm::vec2(0.15f, 0.15f));
	gb5->SetParamsHold((void*)2);
	GUIAction* gb5a = new GUIActionMoveActiveObject(gb5);
	gb5->AddActionHold(gb5a);

	GUIButton* gb6 = new GUIButton(&tb6);
	gb6->Initialize();
	gb6->SetTextures(ResourceManager::GetInstance()->LoadTexture(&tBtnArr), ResourceManager::GetInstance()->LoadTexture(&tBtnArrA));
	gb6->SetPosition(glm::vec2(-0.6f, -0.55f));
	gb6->SetRotation(M_PI * 0.75f);
	gb6->SetScale(glm::vec2(0.15f, 0.15f));
	gb6->SetParamsHold((void*)3);
	GUIAction* gb6a = new GUIActionMoveActiveObject(gb6);
	gb6->AddActionHold(gb6a);

	GUIButton* gb7 = new GUIButton(&tb7);
	gb7->Initialize();
	gb7->SetTextures(ResourceManager::GetInstance()->LoadTexture(&tBtnArr), ResourceManager::GetInstance()->LoadTexture(&tBtnArrA));
	gb7->SetPosition(glm::vec2(-0.2f, -0.55f));
	gb7->SetRotation(M_PI * 0.25f);
	gb7->SetScale(glm::vec2(0.15f, 0.15f));
	gb7->SetParamsHold((void*)4);
	GUIAction* gb7a = new GUIActionMoveActiveObject(gb7);
	gb7->AddActionHold(gb7a);

	GUIButton* gb8 = new GUIButton(&tb8);
	gb8->Initialize();
	gb8->SetTextures(ResourceManager::GetInstance()->LoadTexture(&tBtnArr), ResourceManager::GetInstance()->LoadTexture(&tBtnArrA));
	gb8->SetPosition(glm::vec2(-0.8f, -0.3f));
	gb8->SetRotation(0.0f);
	gb8->SetScale(glm::vec2(0.15f, 0.15f));
	gb8->SetParamsHold((void*)5);
	GUIAction* gb8a = new GUIActionMoveActiveObject(gb8);
	gb8->AddActionHold(gb8a);

	GUIButton* gb9 = new GUIButton(&tb9);
	gb9->Initialize();
	gb9->SetTextures(ResourceManager::GetInstance()->LoadTexture(&tBtnArr), ResourceManager::GetInstance()->LoadTexture(&tBtnArrA));
	gb9->SetPosition(glm::vec2(-0.8f, -0.8f));
	gb9->SetRotation(M_PI * 0.5f);
	gb9->SetScale(glm::vec2(0.15f, 0.15f));
	gb9->SetParamsHold((void*)6);
	GUIAction* gb9a = new GUIActionMoveActiveObject(gb9);
	gb9->AddActionHold(gb9a);

	AddGUIElement(gt);
	AddGUIElement(gt2);
	AddGUIElement(gt3);
	AddGUIElement(gt4);
	AddGUIElement(gt5);
	AddGUIElement(gt6);
	AddGUIElement(gb1);
	AddGUIElement(gb2);
	AddGUIElement(gb3);
	AddGUIElement(gb4);
	AddGUIElement(gb5);
	AddGUIElement(gb6);
	AddGUIElement(gb7);
	AddGUIElement(gb8);
	AddGUIElement(gb9);

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