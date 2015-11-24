#pragma once

/*
	This class controls all input devices and provides an interface for InputHandler to read input device status.
*/

#include "Common.h"
#include "Singleton.h"
#include "Renderer.h"

#include <vector>

class InputManager : public Singleton<InputManager>
{
	friend class Singleton<InputManager>;
	friend class System;

protected:
	InputManager();

	std::vector<int> m_pressed;	// this vector stores pressed buttons WHICH WE HAVE QUERIED
	int m_scrollHelper = 0;

	static int32_t AHandleInput(struct android_app* app, AInputEvent* event);
public:
	InputManager(const InputManager*);
	~InputManager();

	unsigned int Initialize();
	unsigned int Shutdown();
	unsigned int Run();

	bool GetTouch();
	bool GetDoubleTouch();
	bool GetSwipe();
	void GetTouchPosition(glm::vec2* vec);
	void GetDoubleTouchPosition(glm::vec2* vec);
	void GetTouchDirection(glm::vec2* vec);
	void GetDoubleTouchDirection(glm::vec2* vec);
	void GetSwipeDirections(glm::vec4* vec);
};

