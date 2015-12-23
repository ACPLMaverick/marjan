#pragma once
#include "GUIElement.h"

class GUIPicture;
class GUIButton;
class GUIText;
class GUIValueSetter;

class GUISettingsScreen :
	public GUIElement
{
protected:
	GUIPicture* m_bg;
	GUIButton* m_btnApply;
	GUIButton* m_btnCancel;
	GUIText* m_lblSettings;
	GUIValueSetter* m_vsSimType;		// 0 - MS_GPU, 1 - MS_CPU, 2 - PB_GPU, 3 - PB_CPU
	GUIValueSetter* m_vsWidth;			// 1.0f - 50.0f
	GUIValueSetter* m_vsLength;			// 1.0f - 50.0f
	GUIValueSetter* m_vsEdgesWidth;		// 0 - 126
	GUIValueSetter* m_vsEdgesLength;	// 0 - 126
	GUIValueSetter* m_vsElasticity;		// 0.0f - 100.0f
	GUIValueSetter* m_vsMass;			// 1.0f - 100.0f
	GUIValueSetter* m_vsAirDamp;		// 0.0f - 1.0f
	GUIValueSetter* m_vsElDamp;			// 0.0f - 100.0f

public:
	GUISettingsScreen(const std::string* id);
	GUISettingsScreen(const GUISettingsScreen* c);
	~GUISettingsScreen();

	virtual unsigned int Initialize();
	virtual unsigned int Shutdown();
};

