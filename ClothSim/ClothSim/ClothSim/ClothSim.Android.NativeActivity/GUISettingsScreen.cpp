#include "pch.h"
#include "GUISettingsScreen.h"
#include "GUIPicture.h"
#include "GUIButton.h"
#include "GUIValueSetter.h"
#include "GUIText.h"


GUISettingsScreen::GUISettingsScreen(const std::string* id) : GUIElement(id)
{
}

GUISettingsScreen::GUISettingsScreen(const GUISettingsScreen * c) : GUIElement(c)
{
}

GUISettingsScreen::~GUISettingsScreen()
{
}

unsigned int GUISettingsScreen::Initialize()
{
	unsigned int err = CS_ERR_NONE;

	err = GUIElement::Initialize();
	if (err != CS_ERR_NONE)
		return err;

	string vsSimTypeID = m_id + "_vsSimType";
	string vsWidthID = m_id + "_vsWidth";
	string vsLengthID = m_id + "_vsLength";
	string vsEdgesWidthID = m_id + "_vsEdgesWidth";
	string vsEdgesLengthID = m_id + "_vsEdgesLength";
	string vsElasticityID = m_id + "_vsElasticity";
	string vsMassID = m_id + "_vsMass";
	string vsAirDampID = m_id + "_vsAirDamp";
	string vsElDampID = m_id + "_vsElDamp";

	string btnApplyID = m_id + "_btnApply";
	string btnCancelID = m_id + "_btnCancel";
	string lblSettings = m_id + "_lblSettings";
	string bgID = m_id + "_bgID";

	string tPlus = "textures/btn_arrow_up.png";
	string tPlusA = "textures/btn_arrow_up_a.png";
	string tMinus = "textures/btn_arrow_forward.png";
	string tMinusA = "textures/btn_arrow_forward_a.png";
	string tText = "textures/ExportedFont.tga";
	string tBg = "textures/bg_settings.png";

	string vsSimTypeLBL = "Simulation mode";
	string vsWidthLBL = "Width";
	string vsLengthLBL = "Length";
	string vsEdgesWidthLBL = "Edges width";
	string vsEdgesLengthLBL = "Edges length";
	string vsElasticityLBL = "Elasticity";
	string vsMassLBL = "Mass";
	string vsAirDampLBL = "Air damp";
	string vsElDampLBL ="Elasticity damp";

	string mainLBL = "Cloth parameters";

	glm::vec2 txtScale = glm::vec2(0.05, 0.05f);
	glm::vec2 txtPos = glm::vec2(-0.7f, 0.65f);
	glm::vec2 btnScale = glm::vec2(0.2f, 0.2f);
	glm::vec2 btnApplyPos = glm::vec2(0.3f, -0.8f);
	glm::vec2 btnCancelPos = glm::vec2(-0.3f, -0.8f);
	glm::vec2 vsScale = glm::vec2(0.3f, 0.038f);
	glm::vec2 vsSScale = glm::vec2(0.75f, 0.038f);

	int vsCount = 9;
	int inOneCollumn = (int)glm::ceil((float)vsCount / 2.0f);
	float vsXOffset = -0.45f;
	float vsYOffset = 2.0f / (float)vsCount;
	float vsYStart = 0.45f;

	////////////////////////////////

	std::vector<string> vsSimTypeStateLabels;
	vsSimTypeStateLabels.push_back("GPU - MassSpring");
	vsSimTypeStateLabels.push_back("CPU - MassSpring");
	vsSimTypeStateLabels.push_back(" GPU - PosBased");
	vsSimTypeStateLabels.push_back(" CPU - PosBased");
	m_vsSimType = new GUIValueSetter
		(
			&vsSimTypeID,
			&vsSimTypeLBL,
			ResourceManager::GetInstance()->LoadTexture(&tPlus),
			ResourceManager::GetInstance()->LoadTexture(&tPlusA),
			ResourceManager::GetInstance()->LoadTexture(&tMinus),
			ResourceManager::GetInstance()->LoadTexture(&tMinusA),
			ResourceManager::GetInstance()->LoadTexture(&tText),
			4,
			0,
			&vsSimTypeStateLabels,
			-0.3f
		);
	m_vsSimType->SetPosition(m_position + glm::vec2(0.0f, vsYStart));
	m_vsSimType->SetScale(vsSScale);
	m_vsSimType->Initialize();

	m_vsWidth = new GUIValueSetter
		(
			&vsWidthID,
			&vsWidthLBL,
			ResourceManager::GetInstance()->LoadTexture(&tPlus),
			ResourceManager::GetInstance()->LoadTexture(&tPlusA),
			ResourceManager::GetInstance()->LoadTexture(&tMinus),
			ResourceManager::GetInstance()->LoadTexture(&tMinusA),
			ResourceManager::GetInstance()->LoadTexture(&tText),
			50,
			10,
			1.0f,
			1.0f,
			0,
			0.0f
			);

	m_vsLength = new GUIValueSetter
		(
			&vsLengthID,
			&vsLengthLBL,
			ResourceManager::GetInstance()->LoadTexture(&tPlus),
			ResourceManager::GetInstance()->LoadTexture(&tPlusA),
			ResourceManager::GetInstance()->LoadTexture(&tMinus),
			ResourceManager::GetInstance()->LoadTexture(&tMinusA),
			ResourceManager::GetInstance()->LoadTexture(&tText),
			50,
			10,
			1.0f,
			1.0f,
			0,
			0.0f
			);

	m_vsEdgesWidth = new GUIValueSetter
		(
			&vsEdgesWidthID,
			&vsEdgesWidthLBL,
			ResourceManager::GetInstance()->LoadTexture(&tPlus),
			ResourceManager::GetInstance()->LoadTexture(&tPlusA),
			ResourceManager::GetInstance()->LoadTexture(&tMinus),
			ResourceManager::GetInstance()->LoadTexture(&tMinusA),
			ResourceManager::GetInstance()->LoadTexture(&tText),
			126,
			23,
			1.0f,
			0.0f,
			0,
			0.0f
			);

	m_vsEdgesLength = new GUIValueSetter
		(
			&vsEdgesLengthID,
			&vsEdgesLengthLBL,
			ResourceManager::GetInstance()->LoadTexture(&tPlus),
			ResourceManager::GetInstance()->LoadTexture(&tPlusA),
			ResourceManager::GetInstance()->LoadTexture(&tMinus),
			ResourceManager::GetInstance()->LoadTexture(&tMinusA),
			ResourceManager::GetInstance()->LoadTexture(&tText),
			126,
			23,
			1.0f,
			0.0f,
			0,
			0.0f
			);

	m_vsElasticity = new GUIValueSetter
		(
			&vsElasticityID,
			&vsElasticityLBL,
			ResourceManager::GetInstance()->LoadTexture(&tPlus),
			ResourceManager::GetInstance()->LoadTexture(&tPlusA),
			ResourceManager::GetInstance()->LoadTexture(&tMinus),
			ResourceManager::GetInstance()->LoadTexture(&tMinusA),
			ResourceManager::GetInstance()->LoadTexture(&tText),
			101,
			51,
			1.0f,
			0.0f,
			2,
			0.0f
			);

	m_vsMass = new GUIValueSetter
		(
			&vsMassID,
			&vsMassLBL,
			ResourceManager::GetInstance()->LoadTexture(&tPlus),
			ResourceManager::GetInstance()->LoadTexture(&tPlusA),
			ResourceManager::GetInstance()->LoadTexture(&tMinus),
			ResourceManager::GetInstance()->LoadTexture(&tMinusA),
			ResourceManager::GetInstance()->LoadTexture(&tText),
			100,
			1,
			1.0f,
			1.0f,
			2,
			0.0f
			);

	m_vsAirDamp = new GUIValueSetter
		(
			&vsAirDampID,
			&vsAirDampLBL,
			ResourceManager::GetInstance()->LoadTexture(&tPlus),
			ResourceManager::GetInstance()->LoadTexture(&tPlusA),
			ResourceManager::GetInstance()->LoadTexture(&tMinus),
			ResourceManager::GetInstance()->LoadTexture(&tMinusA),
			ResourceManager::GetInstance()->LoadTexture(&tText),
			100,
			99,
			0.01f,
			0.0f,
			2,
			0.0f
			);

	m_vsElDamp = new GUIValueSetter
		(
			&vsElDampID,
			&vsElDampLBL,
			ResourceManager::GetInstance()->LoadTexture(&tPlus),
			ResourceManager::GetInstance()->LoadTexture(&tPlusA),
			ResourceManager::GetInstance()->LoadTexture(&tMinus),
			ResourceManager::GetInstance()->LoadTexture(&tMinusA),
			ResourceManager::GetInstance()->LoadTexture(&tText),
			101,
			100,
			1.0f,
			0.0f,
			2,
			0.0f
			);

	AddChild(m_vsSimType);
	AddChild(m_vsWidth);
	AddChild(m_vsLength);
	AddChild(m_vsEdgesLength);
	AddChild(m_vsEdgesWidth);
	AddChild(m_vsElasticity);
	AddChild(m_vsMass);
	AddChild(m_vsAirDamp);
	AddChild(m_vsElDamp);

	std::vector<GUIValueSetter*> setters;
	setters.push_back(m_vsWidth);
	setters.push_back(m_vsLength);
	setters.push_back(m_vsEdgesLength);
	setters.push_back(m_vsEdgesWidth);
	setters.push_back(m_vsElasticity);
	setters.push_back(m_vsMass);
	setters.push_back(m_vsAirDamp);
	setters.push_back(m_vsElDamp);

	int i = 0;
	for (std::vector<GUIValueSetter*>::iterator it = setters.begin(); it != setters.end(); ++it, ++i)
	{
		GUIValueSetter* vs = *it;
		vs->SetScale(vsScale);
		float xPos, yPos;
		xPos = vsXOffset;
		yPos = vsYStart - vsYOffset * (((i / 2) % inOneCollumn) + 1);
		if (i % 2)
			xPos *= -1.0f;
		vs->SetPosition(m_position + glm::vec2(xPos, yPos));
		vs->Initialize();
	}


	m_bg = new GUIPicture(&bgID, ResourceManager::GetInstance()->LoadTexture(&tBg));
	m_bg->Initialize();
	m_bg->SetScaled(false);
	m_bg->SetPosition(m_position);
	AddChild(m_bg);

	m_lblSettings = new GUIText(&lblSettings, &mainLBL, ResourceManager::GetInstance()->LoadTexture(&tText));
	m_lblSettings->Initialize();
	m_lblSettings->SetScale(txtScale);
	m_lblSettings->SetPosition(txtPos);
	AddChild(m_lblSettings);

	m_btnApply = new GUIButton(&btnApplyID);
	m_btnApply->Initialize();
	m_btnApply->SetScale(btnScale);
	m_btnApply->SetPosition(btnApplyPos);
	m_btnApply->SetTextures(
		ResourceManager::GetInstance()->LoadTexture(&tPlus),
		ResourceManager::GetInstance()->LoadTexture(&tPlusA)
		);
	AddChild(m_btnApply);

	m_btnCancel = new GUIButton(&btnCancelID);
	m_btnCancel->Initialize();
	m_btnCancel->SetScale(btnScale);
	m_btnCancel->SetPosition(btnCancelPos);
	m_btnCancel->SetTextures(
		ResourceManager::GetInstance()->LoadTexture(&tMinus),
		ResourceManager::GetInstance()->LoadTexture(&tMinusA)
		);
	AddChild(m_btnCancel);

	return err;
}

unsigned int GUISettingsScreen::Shutdown()
{
	unsigned int err = CS_ERR_NONE;

	for (std::map<string, GUIElement*>::iterator it = m_children.begin(); it != m_children.end(); ++it)
	{
		it->second->Shutdown();
		delete it->second;
	}
	m_children.clear();

	err = GUIElement::Shutdown();
	if (err != CS_ERR_NONE)
		return err;

	return err;
}
