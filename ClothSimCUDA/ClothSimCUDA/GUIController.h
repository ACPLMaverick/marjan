#pragma once

/*
	This component is the main controller of the whole simulation. Mainly responds to user input.
*/

#include "Common.h"
#include "Component.h"
#include "InputHandler.h"

#include <sstream>
#include <iomanip>

class GUIText;

#define INFO_UPDATE_RATE 40.0
#define BOX_SPEED 0.01f

class GUIController :
	public Component
{
private:
	bool cursorHideHelper = false;
	double infoTimeDisplayHelper = 0.0;

	GUIText* m_fpsText;
	GUIText* m_dtText;
	GUIText* m_ttText;

	void DoubleToStringPrecision(double, int, std::string*);
public:
	GUIController(SimObject* obj);
	GUIController(const GUIController*);
	~GUIController();

	virtual unsigned int Initialize();
	virtual unsigned int Shutdown();

	virtual unsigned int Update();
	virtual unsigned int Draw();
};
