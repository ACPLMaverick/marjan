#include "GUIElement.h"
#include "System.h"

GUIElement::GUIElement(const std::string* id)
{
	m_id = *id;

	m_position = glm::vec2(0.0f, 0.0f);
	m_scale = glm::vec2(1.0f, 1.0f);

	GenerateTransformMatrix();
}

GUIElement::GUIElement(const GUIElement*)
{
}

GUIElement::~GUIElement()
{
}



void GUIElement::GenerateTransformMatrix()
{
	Engine* engine = System::GetInstance()->GetEngineData();
	float scrWidth = engine->width;
	float scrHeight = engine->height;
	float factorX = (scrHeight / scrWidth);
	float factorY = (scrWidth / scrHeight);
	float off = 0.0f;
	if (scrWidth > scrHeight)
	{
		factorY = 1.0f / factorY * 2.0f;
		factorX *= 2.0f;
		off = 0.1f;
	}	

	m_transform = glm::translate(glm::vec3(m_position.x, m_position.y - off, 0.0f)) * glm::scale(glm::vec3(m_scale.x * factorX, m_scale.y * factorY, 0.0f));
}



void GUIElement::SetPosition(glm::vec2 pos)
{
	m_position = pos;
	GenerateTransformMatrix();
}

void GUIElement::SetScale(glm::vec2 scl)
{
	m_scale = scl;
	GenerateTransformMatrix();
}



std::string* GUIElement::GetID()
{
	return &m_id;
}

glm::mat4* GUIElement::GetTransformMatrix()
{
	return &m_transform;
}

glm::vec2 GUIElement::GetPosition()
{
	return m_position;
}

glm::vec2 GUIElement::GetScale()
{
	return m_scale;
}

void GUIElement::FlushDimensions()
{
	GenerateTransformMatrix();
}
