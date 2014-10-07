#ifndef _GRAPHICS_H_
#define _GRAPHICS_H_

//defines
#define DKGRAY_BRUSH 3
#define GRAY_BRUSH 2
//includes
#include <Windows.h>

// globals
const bool FULL_SCREEN = false;
const bool SHOW_CURSOR = false;
const unsigned int BACKGROUND_COLOR = DKGRAY_BRUSH;
const float SCREEN_FAR = 1000.0f;
const float SCREEN_NEAR = 0.1f;

class Graphics
{
private:
	bool Render();
public:
	Graphics();
	Graphics(const Graphics&);
	~Graphics();

	bool Initialize(int, int, HWND);
	void Shutdown();
	bool Frame();
};

#endif

