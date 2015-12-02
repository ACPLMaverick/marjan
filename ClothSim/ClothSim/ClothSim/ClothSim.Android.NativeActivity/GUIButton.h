#pragma once

/*
	This class is a simple clickable button, that an action can be assigned to via function pointer.
*/

#include "GUIElement.h"
#include "GUIAction.h"
#include "InputHandler.h"
#include "MeshGLRectButton.h"

#include <vector>

class GUIAction;
class MeshGLRectButton;

class GUIButton :
	public GUIElement
{
protected:
	std::vector<GUIAction*> m_actionsClick;
	std::vector<GUIAction*> m_actionsHold;

	glm::vec4 m_color;

	TextureID* m_textureIdle;
	TextureID* m_textureClicked;

	MeshGLRectButton* m_mesh;

	void* m_paramsClick;
	void* m_paramsHold;

	
	virtual void GenerateTransformMatrix();
public:
	GUIButton(const std::string* id);
	GUIButton(const GUIButton*);
	~GUIButton();

	virtual unsigned int Initialize();
	virtual unsigned int Shutdown();

	virtual unsigned int Update();
	virtual unsigned int Draw();

	void SetTextures(TextureID* texIdle, TextureID* texClicked);
	void RemoveTextures();

	void AddActionClick(GUIAction* action);
	void AddActionHold(GUIAction* action);

	void RemoveActionClick(GUIAction* action);
	void RemoveActionClick(unsigned int id);
	void RemoveActionHold(GUIAction* action);
	void RemoveActionHold(unsigned int id);

	unsigned int ExecuteActionsClick(void* params);
	unsigned int ExecuteActionsHold(void* params);
};

