#pragma once

/*
	This component is the main controller of the whole simulation. Mainly responds to user input.
*/

#include "Common.h"
#include "Component.h"
#include "InputHandler.h"
#include "Settings.h"

#include <sstream>
#include <iomanip>

class GUIText;
class ClothSimulator;

class GUIController :
	public Component
{
private:
	double infoTimeDisplayHelper = 0.0;

	const float INFO_UPDATE_RATE = 1000.0f;

	GUIText* m_fpsText;
	GUIText* m_dtText;
	GUIText* m_ttText;

	ClothSimulator* m_cSimulator;

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

