#include "pch.h"
#include "GUISlider.h"
#include "GUIText.h"
#include "GUIPicture.h"
#include "GUIActionSetSliderHead.h"
#include "GUIActionMoveSliderHead.h"
#include "GUIButton.h"


GUISlider::GUISlider(const std::string * name, const std::string* label, TextureID * headTex, TextureID * barTex, TextureID * fontTex, 
	unsigned int states, unsigned int defState, float labelMultiplier, float labelOffset) : GUIElement(name)
{
	m_label = *label;

	m_headTex = headTex;
	m_barTex = barTex;
	m_fontTex = fontTex;
	m_states = states;
	m_defState = defState;
	
	m_labelMultiplier = labelMultiplier;
	m_labelOffset = labelOffset;
}

GUISlider::GUISlider(const std::string * name, const std::string* label, TextureID * headTex, TextureID * barTex, TextureID * fontTex, 
	unsigned int states, unsigned int defState, std::vector<string>* labels) : GUISlider(name, label, headTex, barTex, fontTex, states, defState, 0.0f, 0.0f)
{
	size_t size = labels->size();

	if (size != states)
		LOGW("GUISlider: label size is not the same as state count.");

	for (size_t i = 0; i < size && i < (size_t)states; ++i)
	{
		m_valStrings.push_back(labels->at(i));
	}

	if (size < (size_t)states)
	{
		size_t dCount = (size_t)states - size;
		for (size_t i = 0; i < dCount; ++i)
		{
			m_valStrings.push_back("???");
		}
	}

	labelInitialized = true;
}

GUISlider::GUISlider(const GUISlider * c) : GUIElement(c)
{
}

GUISlider::~GUISlider()
{
}

unsigned int GUISlider::Initialize()
{
	unsigned int err = CS_ERR_NONE;

	err = GUIElement::Initialize();
	if (err != CS_ERR_NONE)
		return err;

	if (!labelInitialized)
	{
		for (unsigned int i = 0; i < m_states; ++i)
		{
			float curr = (float)i * m_labelMultiplier + m_labelOffset;
			string str;
			DoubleToStringPrecision(curr, 2, &str);
			m_valStrings.push_back(str);
		}
	}

	float sclFactor = glm::min(m_scale.x, m_scale.y);
	float sclMplier = 2.0f;
	glm::vec2 itemScale = glm::vec2(sclFactor, sclFactor);
	glm::vec2 offsetTl = glm::vec2(-0.6f, 0.2f) * sclFactor;
	glm::vec2 offsetTv = glm::vec2(0.6f, 0.2f) * sclFactor;

	string idTl = m_id + "_TextLabel";
	string idTv = m_id + "_TextVal";
	string idBt = m_id + "_SliderHead";
	string idPc = m_id + "_SliderBar";

	m_textLabel = new GUIText(&idTl, &m_label, m_fontTex);
	m_textLabel->SetScale(itemScale / sclMplier);
	m_textLabel->SetPosition(m_position + offsetTl);
	m_textLabel->Initialize();
	AddChild(m_textLabel);

	m_textValue = new GUIText(&idTv, &(m_valStrings.at(0)), m_fontTex);
	m_textValue->SetScale(itemScale / sclMplier);
	m_textValue->SetPosition(m_position + offsetTv);
	m_textValue->Initialize();
	AddChild(m_textValue);

	m_sliderHead = new GUIButton(&idBt);
	m_sliderHead->Initialize();
	float sp = m_position.x - m_length / 2.0f;
	float step = m_length / (float)(m_states - 1);
	sp += step * ((float)m_defState);
	m_sliderHead->SetPosition(glm::vec2(sp, m_position.y));
	m_sliderHead->SetScale(itemScale * sclMplier);
	m_sliderHead->SetTextures(m_headTex, m_headTex);
	m_actionMove = new GUIActionMoveSliderHead(m_sliderHead, m_length);
	m_sliderHead->AddActionHold(m_actionMove);
	
	m_actionMove->Initialize();
	m_actionSet = new GUIActionSetSliderHead(m_sliderHead, m_length, m_states, m_defState);
	m_sliderHead->AddActionClick(m_actionSet);
	m_actionSet->Initialize();
	AddChild(m_sliderHead);

	m_sliderBar = new GUIPicture(&idPc, m_barTex);
	m_sliderBar->Initialize();
	m_sliderBar->SetPosition(m_position);
	m_sliderBar->SetScale(m_scale * sclMplier);
	AddChild(m_sliderBar);

	for (std::map<string, GUIElement*>::iterator it = m_children.begin(); it != m_children.end(); ++it)
	{
		(*it).second->SetBlockable(true);
	}

	return err;
}

unsigned int GUISlider::Shutdown()
{
	unsigned int err = CS_ERR_NONE;

	err = GUIElement::Shutdown();
	if (err != CS_ERR_NONE)
		return err;

	m_textLabel->Shutdown();
	m_textValue->Shutdown();
	m_sliderBar->Shutdown();
	m_sliderHead->Shutdown();
	delete m_textLabel;
	delete m_textValue;
	delete m_sliderBar;
	delete m_sliderHead;

	return err;
}

unsigned int GUISlider::Update()
{
	unsigned int err = CS_ERR_NONE;

	if (m_isEnabled)
	{
		err = GUIElement::Update();
		if (err != CS_ERR_NONE)
			return err;

		unsigned int temp = m_actionSet->GetCurrentOption();
		if (temp != m_currentState)
		{
			m_currentState = temp;
			for (std::vector<std::function<void(unsigned int)>>::iterator it = EventStateChanged.begin(); it != EventStateChanged.end(); ++it)
			{
				(*it)(m_currentState);
			}
		}
	}

	return err;
}
