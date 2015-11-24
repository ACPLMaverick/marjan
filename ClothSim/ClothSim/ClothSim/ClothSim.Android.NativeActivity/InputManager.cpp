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

	

	return err;
}

bool InputManager::GetTouch()
{
	return false;
}

bool InputManager::GetDoubleTouch()
{
	return false;
}

bool InputManager::GetSwipe()
{
	return false;
}

void InputManager::GetTouchPosition(glm::vec2 * vec)
{
}

void InputManager::GetDoubleTouchPosition(glm::vec2 * vec)
{
}

void InputManager::GetTouchDirection(glm::vec2 * vec)
{
}

void InputManager::GetDoubleTouchDirection(glm::vec2 * vec)
{
}

void InputManager::GetSwipeDirections(glm::vec4 * vec)
{
}

/**
* Process the next input event.
*/
int32_t InputManager::AHandleInput(struct android_app* app, AInputEvent* event) {
	Engine* engine = System::GetInstance()->GetEngineData();

	if (AInputEvent_getType(event) == AINPUT_EVENT_TYPE_MOTION) {
		engine->state.x = AMotionEvent_getX(event, 0);
		engine->state.y = AMotionEvent_getY(event, 0);
		return 1;
	}
	
	return 0;
}