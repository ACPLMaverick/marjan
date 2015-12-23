#pragma once

/*
	This component is the main controller of the whole simulation. Mainly responds to user input.
*/

#include "Common.h"
#include "Component.h"
#include "InputHandler.h"
#include "Settings.h"

class GUIText;
class ClothSimulator;

class GUIController :
	public Component
{
private:
	enum MovementDirection
	{
		FORWARD = 1,
		BACKWARD,
		LEFT,
		RIGHT,
		UP,
		DOWN
	};

	double infoTimeDisplayHelper = 0.0;

	static constexpr float INFO_UPDATE_RATE = 1000.0f;
	static constexpr float BOX_SPEED = 0.005f;

	GUIText* m_fpsText;
	GUIText* m_dtText;
	GUIText* m_ttText;

	ClothSimulator* m_cSimulator;

	static void ActionExitProgram(std::vector<void*>* params, const glm::vec2* clickPos);
	static void ActionMoveActiveObject(std::vector<void*>* params, const glm::vec2* clickPos);
	static void ActionSetDisplayMode(std::vector<void*>* params, const glm::vec2* clickPos);
	static void ActionShowPreferences(std::vector<void*>* params, const glm::vec2* clickPos);
	static void ActionApplyPreferences(std::vector<void*>* params, const glm::vec2* clickPos);
	static void ActionCancelPreferences(std::vector<void*>* params, const glm::vec2* clickPos);
public:
	GUIController(SimObject* obj);
	GUIController(const GUIController*);
	~GUIController();

	virtual unsigned int Initialize();
	virtual unsigned int Shutdown();

	virtual unsigned int Update();
	virtual unsigned int Draw();
};

