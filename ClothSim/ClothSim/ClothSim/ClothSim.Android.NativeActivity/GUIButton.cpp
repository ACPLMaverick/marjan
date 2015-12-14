#include "GUIButton.h"


GUIButton::GUIButton(const std::string* id) : GUIElement(id)
{
	m_color = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
	m_rotation = 0.0f;
}

GUIButton::GUIButton(const GUIButton* c) : GUIElement(c)
{
}

GUIButton::~GUIButton()
{
}



unsigned int GUIButton::Initialize()
{
	m_mesh = new MeshGLRectButton(nullptr, this, &m_color);
	m_mesh->Initialize();

	InputManager::GetInstance()->AddButton(this);

	return CS_ERR_NONE;
}

unsigned int GUIButton::Shutdown()
{
	for (std::vector<GUIAction*>::iterator it = m_actionsClick.begin(); it != m_actionsClick.end(); ++it)
	{
		delete (*it);
	}
	m_actionsClick.clear();

	for (std::vector<GUIAction*>::iterator it = m_actionsHold.begin(); it != m_actionsHold.end(); ++it)
	{
		delete (*it);
	}
	m_actionsHold.clear();

	m_mesh->Shutdown();
	delete m_mesh;

	return CS_ERR_NONE;
}



unsigned int GUIButton::Update()
{
	unsigned int err = CS_ERR_NONE;

	/*
	bool isClick = InputHandler::GetInstance()->GetClick();
	bool isHold = InputHandler::GetInstance()->GetHold();

	if (isClick || isHold)
	{
		// we have an event here, so we calculate current finger position
		glm::vec2 clickPos;
		InputHandler::GetInstance()->GetClickPosition(&clickPos);

		Engine* e = System::GetInstance()->GetEngineData();
		float w = e->width;
		float h = e->height;

		clickPos.x = clickPos.x / w * 2.0f - 1.0f;
		clickPos.y = clickPos.y / h * 2.0f - 1.0f;

		glm::vec2 nScl = m_scale;
		glm::vec2 scaleFactors;
		ComputeScaleFactors(&scaleFactors);
		nScl.x *= scaleFactors.x;
		nScl.y *= scaleFactors.y;

		// check if click is within boundaries of this button
		if (
			clickPos.x >= m_position.x - nScl.x &&
			clickPos.x <= m_position.x + nScl.x &&
			clickPos.y >= m_position.y - nScl.y &&
			clickPos.y <= m_position.y + nScl.y
			)
		{
			if (!isClickInProgress)
			{
				isClickInProgress = true;
				m_mesh->SetTextureID(m_textureClicked);
			}

			// YES! so, proceed accordingly
			if (isClick)
				ExecuteActionsClick(m_paramsClick);
			else if (isHold)
				ExecuteActionsHold(m_paramsHold);
		}
		else if (isClickInProgress)
		{
			isClickInProgress = false;
			m_mesh->SetTextureID(m_textureIdle);
		}
	}
	else if (isClickInProgress)
	{
		isClickInProgress = false;
		m_mesh->SetTextureID(m_textureIdle);
	}
	*/

	return err;
}

unsigned int GUIButton::Draw()
{
	unsigned int err = CS_ERR_NONE;

	err = m_mesh->Draw();

	return CS_ERR_NONE;
}

void GUIButton::GenerateTransformMatrix()
{
	glm::vec2 factors;
	ComputeScaleFactors(&factors);

	m_transform = glm::translate(glm::vec3(m_position.x, m_position.y, 0.0f))
		* glm::scale(glm::vec3(m_scale.x * factors.x, m_scale.y * factors.y, 0.0f));

	m_rot = glm::rotate(2.0f * m_rotation, glm::vec3(0.0f, 0.0f, 1.0f));
}

void GUIButton::ComputeScaleFactors(glm::vec2 * factors)
{
	Engine* engine = System::GetInstance()->GetEngineData();
	float scrWidth = engine->width;
	float scrHeight = engine->height;
	float bBias = 0.6f;
	float factorX = (scrHeight / scrWidth);
	float factorY = (scrWidth / scrHeight);
	float hsFactor = 1.0f / Renderer::GetInstance()->GetScreenRatio();
	if (scrWidth > scrHeight)
	{
		factorY = 1.0f / factorY * hsFactor;
		factorX *= hsFactor;
	}

	factorX *= (1.0f / hsFactor);

	factors->x = factorX;
	factors->y = factorY;
}

void GUIButton::SetTextures(TextureID * texIdle, TextureID* texClicked)
{
	m_textureIdle = texIdle;
	m_textureClicked = texClicked;

	m_mesh->SetTextureID(m_textureIdle);
}

void GUIButton::RemoveTextures()
{
	m_textureClicked = ResourceManager::GetInstance()->GetTextureWhite();
	m_textureIdle = ResourceManager::GetInstance()->GetTextureWhite();

	m_mesh->SetTextureID(m_textureIdle);
}

void GUIButton::SetRotation(float r)
{
	m_rotation = r;
}

void GUIButton::SetParamsClick(void * params)
{
	m_paramsClick.push_back(params);
}

void GUIButton::SetParamsHold(void * params)
{
	m_paramsHold.push_back(params);
}

std::vector<void*>* GUIButton::GetParamsClick()
{
	return &m_paramsClick;
}

std::vector<void*>* GUIButton::GetParamsHold()
{
	return &m_paramsHold;
}

float GUIButton::GetRotation()
{
	return m_rotation;
}

glm::mat4 GUIButton::GetRotationMatrix()
{
	return m_rot;
}

void GUIButton::AddActionClick(GUIAction * action)
{
	m_actionsClick.push_back(action);
}

void GUIButton::AddActionHold(GUIAction * action)
{
	m_actionsHold.push_back(action);
}

void GUIButton::RemoveActionClick(GUIAction * action)
{
	for (std::vector<GUIAction*>::iterator it = m_actionsClick.begin(); it != m_actionsClick.end(); ++it)
	{
		if (action == (*it))
		{
			m_actionsClick.erase(it);
			break;
		}
	}
}

void GUIButton::RemoveActionClick(unsigned int id)
{
	unsigned int ctr = 0;
	for (std::vector<GUIAction*>::iterator it = m_actionsClick.begin(); it != m_actionsClick.end(); ++it, ++ctr)
	{
		if (id == ctr)
		{
			m_actionsClick.erase(it);
			break;
		}
	}
}

void GUIButton::RemoveActionHold(GUIAction * action)
{
	for (std::vector<GUIAction*>::iterator it = m_actionsHold.begin(); it != m_actionsHold.end(); ++it)
	{
		if (action == (*it))
		{
			m_actionsHold.erase(it);
			break;
		}
	}
}

void GUIButton::RemoveActionHold(unsigned int id)
{
	unsigned int ctr = 0;
	for (std::vector<GUIAction*>::iterator it = m_actionsHold.begin(); it != m_actionsHold.end(); ++it, ++ctr)
	{
		if (id == ctr)
		{
			m_actionsHold.erase(it);
			break;
		}
	}
}

unsigned int GUIButton::ExecuteActionsClick()
{
	unsigned int err = CS_ERR_NONE;

	for (std::vector<GUIAction*>::iterator it = m_actionsClick.begin(); it != m_actionsClick.end(); ++it)
	{
		(*it)->Action(&m_paramsClick);
	}

	return err;
}

unsigned int GUIButton::ExecuteActionsHold()
{
	unsigned int err = CS_ERR_NONE;

	if (!isClickInProgress)
	{
		isClickInProgress = true;
		m_mesh->SetTextureID(m_textureClicked);
	}

	for (std::vector<GUIAction*>::iterator it = m_actionsHold.begin(); it != m_actionsHold.end(); ++it)
	{
		(*it)->Action(&m_paramsHold);
	}

	return err;
}

void GUIButton::CleanupAfterHold()
{
	if (isClickInProgress)
	{
		isClickInProgress = false;
		m_mesh->SetTextureID(m_textureIdle);
	}
}

bool GUIButton::GetHoldInProgress()
{
	return isClickInProgress;
}
