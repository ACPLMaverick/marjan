#include "pch.h"
#include "GUIActionShowPreferences.h"
#include "ClothSimulator.h"
#include "GUIText.h"

GUIActionShowPreferences::GUIActionShowPreferences(GUIButton* b) : GUIAction(b)
{
}

GUIActionShowPreferences::GUIActionShowPreferences(const GUIActionShowPreferences * c) : GUIAction(c)
{
}


GUIActionShowPreferences::~GUIActionShowPreferences()
{
}

unsigned int GUIActionShowPreferences::Initialize()
{
	unsigned int err = CS_ERR_NONE;

	std::vector<void*>* params = m_button->GetParamsClick();
	ClothSimulator* cSim = (ClothSimulator*)(params->at(0));

	ClothSimulationMode cMode = cSim->GetMode();
	GUIText* txt = (GUIText*)(params->at(1));
	switch (cMode)
	{
	case ClothSimulationMode::MASS_SPRING:
		txt->SetText(&MS_VALUE);
		break;

	case ClothSimulationMode::POSITION_BASED:
		txt->SetText(&PB_VALUE);
		break;

	default:
		txt->SetText(&UN_VALUE);
		break;
	}

	return err;
}

unsigned int GUIActionShowPreferences::Action(std::vector<void*>* params, const glm::vec2* clickPos)
{
	unsigned int err = CS_ERR_NONE;

	if (params->size() != 2)
		return CS_ERR_ACTION_BAD_PARAM;

	/////// temporary

	ClothSimulator* cSim = (ClothSimulator*)(params->at(0));
	cSim->SwitchMode();

	ClothSimulationMode cMode = cSim->GetMode();
	GUIText* txt = (GUIText*)(params->at(1));
	switch (cMode)
	{
		case ClothSimulationMode::MASS_SPRING:
			txt->SetText(&MS_VALUE);
			break;

		case ClothSimulationMode::POSITION_BASED:
			txt->SetText(&PB_VALUE);
			break;

		default:
			txt->SetText(&UN_VALUE);
			break;
	}

	/////////////////

	return err;
}
