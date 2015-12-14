#pragma once

/*
	This class is a simple clickable button, that an action can be assigned to via function pointer.
*/

#include "GUIElement.h"
#include "GUIAction.h"
#include "InputHandler.h"
#include "InputManager.h"
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

	glm::mat4 m_rot;

	glm::vec4 m_color;
	float m_rotation;

	TextureID* m_textureIdle;
	TextureID* m_textureClicked;

	MeshGLRectButton* m_mesh;

	std::vector<void*> m_paramsClick;
	std::vector<void*> m_paramsHold;

	bool isClickInProgress;
	
	virtual void GenerateTransformMatrix();
	void ComputeScaleFactors(glm::vec2* factors);
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

	void SetRotation(float r);
	void SetParamsClick(void* params);
	void SetParamsHold(void* params);
	std::vector<void*>* GetParamsClick();
	std::vector<void*>* GetParamsHold();
	float GetRotation();
	glm::mat4 GetRotationMatrix();

	void AddActionClick(GUIAction* action);
	void AddActionHold(GUIAction* action);

	void RemoveActionClick(GUIAction* action);
	void RemoveActionClick(unsigned int id);
	void RemoveActionHold(GUIAction* action);
	void RemoveActionHold(unsigned int id);

	unsigned int ExecuteActionsClick();
	unsigned int ExecuteActionsHold();
	void CleanupAfterHold();

	bool GetHoldInProgress();
};

