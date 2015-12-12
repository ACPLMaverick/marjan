#include "InputManager.h"
#include "GUIButton.h"

InputManager::InputManager()
{

}

InputManager::InputManager(const InputManager*)
{

}

InputManager::~InputManager()
{

}


unsigned int InputManager::Initialize()
{
	unsigned int err = CS_ERR_NONE;

	m_tBools.push_back(&m_clickHelperTBool);
	m_isClick = false;
	m_isMove = false;
	m_isHold = false;
	m_isHoldDouble = false;
	m_isPinch = false;

	return err;
}

unsigned int InputManager::Shutdown()
{
	unsigned int err = CS_ERR_NONE;

	m_buttons.clear();

	return err;
}

unsigned int InputManager::Run()
{
	unsigned int err = CS_ERR_NONE;

	for (std::vector<TwoBool*>::iterator it = m_tBools.begin(); it != m_tBools.end(); ++it)
	{
		(*it)->Update();
	}

	if (m_clickHelperTBool.GetVal())
	{
		m_isClick = true;
		//m_pressTBool.SetVal(false);
	}
	else
		m_isClick = false;

	//////////////////////////////////////

	if (m_isHold)
		m_currentlyHeldButtons = ProcessButtonHolds(&m_touch01Position);
	else
		m_currentlyHeldButtons = 0;

	return err;
}

bool InputManager::GetPress()
{
	return m_isClick;
}

bool InputManager::GetTouch()
{
	return m_isHold;
}

bool InputManager::GetDoubleTouch()
{
	return m_isHoldDouble;
}

bool InputManager::GetPinch()
{
	return m_isPinch;
}

bool InputManager::GetMove()
{
	return m_isMove;
}

void InputManager::GetTouchPosition(glm::vec2 * vec)
{
	vec->x = m_touch01Position.x;
	vec->y = m_touch01Position.y;
}

void InputManager::GetDoubleTouchPosition(glm::vec2 * vec)
{
	vec->x = (m_touch01Position.x + m_touch02Position.x) / 2.0f;
	vec->y = (m_touch01Position.y + m_touch02Position.y) / 2.0f;
}

void InputManager::GetTouchDirection(glm::vec2 * vec)
{
	vec->x = m_touch01Direction.x;
	vec->y = m_touch01Direction.y;
}

void InputManager::GetDoubleTouchDirection(glm::vec2 * vec)
{
	vec->x = (m_touch01Direction.x + m_touch02Direction.x) / 2.0f;
	vec->y = (m_touch01Direction.y + m_touch02Direction.y) / 2.0f;
}

float InputManager::GetPinchValue()
{
	return m_pinchVal;
}

unsigned int InputManager::GetCurrentlyHeldButtons()
{
	return m_currentlyHeldButtons;
}

void InputManager::AddButton(GUIButton * button)
{
	m_buttons.push_back(button);
}

void InputManager::RemoveButton(GUIButton * button)
{
	int ctr = 0;
	for (std::vector<GUIButton*>::iterator it = m_buttons.begin(); it != m_buttons.end(); ++it, ++ctr)
	{
		if ((*it) == button)
		{
			m_buttons.erase(it);
			return;
		}
	}
}

void InputManager::ComputeScaleFactors(glm::vec2 * factors)
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

bool InputManager::ButtonAreaInClick(GUIButton * button, const glm::vec2 * clickPos)
{
	// we have an event here, so we calculate current finger position
	glm::vec2 cp;

	Engine* e = System::GetInstance()->GetEngineData();
	float w = e->width;
	float h = e->height;

	cp.x = clickPos->x / w * 2.0f - 1.0f;
	cp.y = -(clickPos->y / h * 2.0f - 1.0f);

	glm::vec2 nScl = button->GetScale();
	glm::vec2 pos = button->GetPosition();
	glm::vec2 scaleFactors;
	ComputeScaleFactors(&scaleFactors);
	nScl.x *= scaleFactors.x;
	nScl.y *= scaleFactors.y;

	// check if click is within boundaries of this button
	if (
		cp.x >= pos.x - nScl.x &&
		cp.x <= pos.x + nScl.x &&
		cp.y >= pos.y - nScl.y &&
		cp.y <= pos.y + nScl.y
		)
	{
		return true;
	}

	return false;
}

unsigned int InputManager::ProcessButtonClicks(const glm::vec2 * clickPos)
{
	unsigned int ctr = 0;
	for (std::vector<GUIButton*>::iterator it = m_buttons.begin(); it != m_buttons.end(); ++it)
	{
		if (ButtonAreaInClick(*it, clickPos))
		{
			++ctr;
			(*it)->ExecuteActionsClick();
			(*it)->CleanupAfterHold();
		}
	}

	return ctr;
}

unsigned int InputManager::ProcessButtonHolds(const glm::vec2 * clickPos)
{
	unsigned int ctr = 0;
	for (std::vector<GUIButton*>::iterator it = m_buttons.begin(); it != m_buttons.end(); ++it)
	{
		if (ButtonAreaInClick(*it, clickPos))
		{
			++ctr;
			(*it)->ExecuteActionsHold();
		}
		else if ((*it)->GetHoldInProgress())
		{
			(*it)->CleanupAfterHold();
		}
	}
	return ctr;
}

/**
* Process the next input event.
*/
int32_t InputManager::AHandleInput(struct android_app* app, AInputEvent* event) 
{
	Engine* engine = System::GetInstance()->GetEngineData();

	if (AInputEvent_getType(event) == AINPUT_EVENT_TYPE_MOTION &&
		AInputEvent_getSource(event) == AINPUT_SOURCE_TOUCHSCREEN
		) 
	{
		//LOGI("InputManager: TOUCHSCREEN! Count: %d", AMotionEvent_getPointerCount(event));
		unsigned int pCount = AMotionEvent_getPointerCount(event);
		InputManager* im = InputManager::GetInstance();

		if (AMotionEvent_getAction(event) == AMOTION_EVENT_ACTION_MOVE)
		{
			im->m_isMove = true;
		}

		if (pCount == 1)
		{
			glm::vec2 tVec;
			
			tVec.x = AMotionEvent_getX(event, 0);
			tVec.y = AMotionEvent_getY(event, 0);

			im->m_touch01Direction = tVec - im->m_touch01Position;
			im->m_touch01Position = tVec;


			if (AMotionEvent_getAction(event) == AMOTION_EVENT_ACTION_UP)
			{
				im->ProcessButtonClicks(&tVec);
				im->m_isHold = false;
			}
			else if(AMotionEvent_getAction(event) == AMOTION_EVENT_ACTION_DOWN)
			{
				im->m_isHold = true;
			}
		}

		else if (pCount == 2)
		{
			glm::vec2 tVec1, tVec2;

			tVec1.x = AMotionEvent_getX(event, 0);
			tVec1.y = AMotionEvent_getY(event, 0);
			tVec2.x = AMotionEvent_getX(event, 1);
			tVec2.y = AMotionEvent_getY(event, 1);

			im->m_touch01Direction = tVec1 - im->m_touch01Position;
			im->m_touch02Direction = tVec2 - im->m_touch02Position;
			im->m_touch01Position = tVec1;
			im->m_touch02Position = tVec2;

			if (AMotionEvent_getAction(event) == AMOTION_EVENT_ACTION_MOVE)
			{
				im->m_isHold = false;
				float dot = glm::dot(im->m_touch01Direction, im->m_touch02Direction);
				if (dot < 0.7f && im->m_isHoldDouble)
				{
					float fl = glm::length(im->m_touch01Direction);
					float sl = glm::length(im->m_touch02Direction);
					float mp = 1.0f;
					float diff = glm::length(im->m_touch01Position - im->m_touch02Position);
					if (diff - im->m_diffPinch < 0)
						mp *= -1.0f;
					im->m_diffPinch = diff;
					im->m_pinchVal = (fl + sl) / 2.0f * mp;

					im->m_isHoldDouble = false;
					im->m_isPinch = true;
				}
				else
				{
					im->m_isHoldDouble = true;
					im->m_isPinch = false;
				}
			}
		}

		// cleanup
		if (AMotionEvent_getAction(event) == AMOTION_EVENT_ACTION_UP)
		{
			im->m_isHold = false;
			im->m_isHoldDouble = false;
			im->m_isPinch = false;
			im->m_isMove = false;
		}
		return 1;
	}
	if (AInputEvent_getType(event) == AINPUT_EVENT_TYPE_KEY)
	{
		// do nothing atm
		return 1;
	}
	
	return 0;
}