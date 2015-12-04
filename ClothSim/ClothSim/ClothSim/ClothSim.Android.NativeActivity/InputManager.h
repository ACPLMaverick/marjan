#pragma once

/*
	This class controls all input devices and provides an interface for InputHandler to read input device status.
*/

#include "Common.h"
#include "Singleton.h"
#include "Renderer.h"

#include <vector>

/***
	This class behaves as a common bool object. Difference is that it's value changes after the second update call.
*/

class GUIButton;


////////////////////////////////////////////

class TwoBool
{
private:
	bool second;
	bool val;
public:
	TwoBool()
	{
		val = false;
	}

	TwoBool(bool value)
	{
		SetVal(value);
	}

	bool GetVal() { return val; }

	void SetVal(bool value)
	{
		val = value;
		second = !value;
	}

	void Update()
	{
		if (val == false && second == false)
			return;

		if (val != second)
		{
			second = val;
			return;
		}
		if (val == second)
		{
			val = false;
			second = false;
			return;
		}
	}
};

///////////////////////////

class InputManager : public Singleton<InputManager>
{
	friend class Singleton<InputManager>;
protected:
	std::vector<GUIButton*> m_buttons;	// this vector stores buttons that recieve DOWN, MOVE and UP events
	int m_scrollHelper = 0;

	TwoBool m_touch01TBool;
	TwoBool m_touch02TBool;
	TwoBool m_pressTBool;
	TwoBool m_clickHelperTBool;
	TwoBool m_doubleTouchTBool;
	TwoBool m_pinchTBool;
	std::vector<TwoBool*> m_tBools;

	glm::vec2 m_touch01Position;
	glm::vec2 m_touch01Direction;
	glm::vec2 m_touch02Position;
	glm::vec2 m_touch02Direction;
	float m_diffPinch;

	float m_pinchVal = 0.0f;
	bool m_isPinch;
	bool m_isClick;
	bool m_isHold;
	unsigned int m_currentlyHeldButtons;

	InputManager();
	inline bool ButtonAreaInClick(GUIButton* button, const glm::vec2* clickPos);
	inline unsigned int ProcessButtonClicks(const glm::vec2 * clickPos);
	inline unsigned int ProcessButtonHolds(const glm::vec2 * clickPos);
	inline void ComputeScaleFactors(glm::vec2* factors);
public:
	InputManager(const InputManager*);
	~InputManager();

	unsigned int Initialize();
	unsigned int Shutdown();
	unsigned int Run();

	bool GetPress();
	bool GetTouch();
	bool GetDoubleTouch();
	bool GetPinch();
	void GetTouchPosition(glm::vec2* vec);
	void GetDoubleTouchPosition(glm::vec2* vec);
	void GetTouchDirection(glm::vec2* vec);
	void GetDoubleTouchDirection(glm::vec2* vec);
	float GetPinchValue();
	unsigned int GetCurrentlyHeldButtons();

	void AddButton(GUIButton* button);
	void RemoveButton(GUIButton* button);

	static int32_t AHandleInput(struct android_app* app, AInputEvent* event);
};

