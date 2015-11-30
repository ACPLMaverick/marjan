#pragma once

/*
*/

#include "Common.h"
#include "Singleton.h"
#include "InputManager.h"

#include <glm\glm\glm.hpp>

class InputHandler : public Singleton<InputHandler>
{
	friend class Singleton<InputHandler>;

protected:
	InputHandler();
public:
	InputHandler(const InputHandler*);
	~InputHandler();

	bool ExitPressed();
	void GetCameraMovementVector(glm::vec2* vec);
	void GetCameraRotationVector(glm::vec2* vec);
	float GetZoomValue();
	bool GetZoom();
	bool ActionButtonPressed();
	bool ActionButtonClicked();
	bool CameraRotateButtonPressed();
	bool CameraMoveButtonPressed();
	bool WireframeButtonClicked();
};

