#pragma once

//***
// The purpose of this class is to govern all computations that take place within the program, 
// which consist of graphic rendering, CUDA cloth computations and input handling
// main loop is implemented here.
//
// This is a singleton class.
//

#include "Common.h"

#include "Renderer.h"
#include "InputManager.h"
#include "InputHandler.h"
#include "Scene.h"
#include "SceneTest.h"
#include "SceneSim.h"

class Renderer;

class System
{
private:
	static System* instance;
	System();

	bool m_running;

	Scene* m_scene;
public:
	System(const System*);
	~System();

	static System* GetInstance();
	static void DestroyInstance();

	unsigned int Initialize();
	unsigned int Shutdown();
	unsigned int Run();
	void Stop();

	Scene* GetCurrentScene();
};

