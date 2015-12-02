#include "GUIButton.h"


GUIButton::GUIButton(const std::string* id) : GUIElement(id)
{
	m_color = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
}

GUIButton::GUIButton(const GUIButton* c) : GUIElement(c)
{
}

GUIButton::~GUIButton()
{
}



unsigned int GUIButton::Initialize()
{
	m_paramsClick = nullptr;
	m_paramsHold = nullptr;

	m_mesh = new MeshGLRectButton(nullptr, this, &m_color);
	m_mesh->Initialize();

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

	bool isClick = InputHandler::GetInstance()->GetPressed();
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

		//LOGI("GUIButton: %f %f", clickPos.x, clickPos.y);

		// check if click is within boundaries of this button
		if (
			clickPos.x >= m_position.x - m_scale.x &&
			clickPos.x <= m_position.x + m_scale.x &&
			clickPos.y >= m_position.y - m_scale.y &&
			clickPos.y <= m_position.y + m_scale.y
			)
		{
			// YES! so, proceed accordingly
			if (isClick)
				ExecuteActionsClick(m_paramsClick);
			else if (isHold)
				ExecuteActionsHold(m_paramsHold);
		}
	}
	

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
	Engine* engine = System::GetInstance()->GetEngineData();
	float scrWidth = engine->width;
	float scrHeight = engine->height;
	float factorX = (scrHeight / scrWidth);
	float factorY = (scrWidth / scrHeight);
	if (scrWidth > scrHeight)
	{
		factorY = 1.0f / factorY * 2.0f;
		factorX *= 2.0f;
	}

	m_transform = glm::translate(glm::vec3(m_position.x, m_position.y, 0.0f)) * glm::scale(glm::vec3(m_scale.x * factorX, m_scale.y * factorY, 0.0f));
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

unsigned int GUIButton::ExecuteActionsClick(void * params)
{
	unsigned int err = CS_ERR_NONE;

	//LOGI("Button: Click!");

	for (std::vector<GUIAction*>::iterator it = m_actionsClick.begin(); it != m_actionsClick.end(); ++it)
	{
		(*it)->Action(params);
	}

	return err;
}

unsigned int GUIButton::ExecuteActionsHold(void * params)
{
	unsigned int err = CS_ERR_NONE;

	//LOGI("Button: Hold!");

	for (std::vector<GUIAction*>::iterator it = m_actionsHold.begin(); it != m_actionsHold.end(); ++it)
	{
		(*it)->Action(params);
	}

	return err;
}
