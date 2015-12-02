#include "InputManager.h"

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

	m_tBools.push_back(&m_pressTBool);
	m_tBools.push_back(&m_clickHelperTBool);
	m_tBools.push_back(&m_touch01TBool);
	m_tBools.push_back(&m_touch02TBool);
	m_tBools.push_back(&m_doubleTouchTBool);
	m_tBools.push_back(&m_pinchTBool);

	return err;
}

unsigned int InputManager::Shutdown()
{
	unsigned int err = CS_ERR_NONE;

	return err;
}

unsigned int InputManager::Run()
{
	unsigned int err = CS_ERR_NONE;

	for (std::vector<TwoBool*>::iterator it = m_tBools.begin(); it != m_tBools.end(); ++it)
	{
		(*it)->Update();
	}

	if (m_clickHelperTBool.GetVal() && !m_pressTBool.GetVal())
	{
		m_isClick = true;
		//m_pressTBool.SetVal(false);
	}
	else
		m_isClick = false;

	m_clickHelperTBool.SetVal(m_pressTBool.GetVal());

	return err;
}

bool InputManager::GetPress()
{
	return m_isClick;
}

bool InputManager::GetTouch()
{
	return m_touch01TBool.GetVal();
}

bool InputManager::GetDoubleTouch()
{
	return m_doubleTouchTBool.GetVal();
}

bool InputManager::GetPinch()
{
	return m_pinchTBool.GetVal();
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

		if (pCount == 1)
		{
			glm::vec2 tVec;
			
			tVec.x = AMotionEvent_getX(event, 0);
			tVec.y = AMotionEvent_getY(event, 0);

			if (im->m_touch01TBool.GetVal())
				im->m_touch01Direction = tVec - im->m_touch01Position;
			im->m_touch01Position = tVec;

			im->m_touch01TBool.SetVal(true);
			im->m_pressTBool.SetVal(true);
		}
		else if (pCount == 2)
		{
			glm::vec2 tVec1, tVec2;

			tVec1.x = AMotionEvent_getX(event, 0);
			tVec1.y = AMotionEvent_getY(event, 0);
			tVec2.x = AMotionEvent_getX(event, 1);
			tVec2.y = AMotionEvent_getY(event, 1);

			if (im->m_doubleTouchTBool.GetVal() || im->m_pinchTBool.GetVal())
			{
				im->m_touch01Direction = tVec1 - im->m_touch01Position;
				im->m_touch02Direction = tVec2 - im->m_touch02Position;

				float dot = glm::dot(im->m_touch01Direction, im->m_touch02Direction);

				// check if we have a pinch action here
				if (dot < 0.7f)
				{
					im->m_pinchTBool.SetVal(true);
					im->m_isPinch = true;

					float fl = glm::length(im->m_touch01Direction);
					float sl = glm::length(im->m_touch02Direction);
					float mp = 1.0f;
					
					float diff = glm::length(im->m_touch01Position - im->m_touch02Position);

					if (diff - im->m_diffPinch < 0)
						mp *= -1.0f;

					im->m_diffPinch = diff;

					im->m_pinchVal = (fl + sl) / 2.0f * mp;
				}
			}
			im->m_touch01Position = tVec1;
			im->m_touch02Position = tVec2;

			im->m_doubleTouchTBool.SetVal(true);
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