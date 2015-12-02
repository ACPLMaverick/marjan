#pragma once

//***
// The purpose of this class is to govern all computations that take place within the program, 
// which consist of graphics rendering, cloth computations and input handling.
// main loop is implemented here.
//
// This is a singleton class.
//

#include "Common.h"
#include "Singleton.h"

#include "Renderer.h"
#include "InputManager.h"
#include "InputHandler.h"
#include "PhysicsManager.h"
#include "Scene.h"
#include "SceneTest.h"
#include "SceneSim.h"
#include "Timer.h"

class Renderer;

///////////////////

/**
* Our saved state data.
* Here will remain all data needed to save and restore our project.
*/
struct saved_state {
	float angle;
	int32_t x;
	int32_t y;
};

/**
* Shared state for our app.
*/
struct Engine {
	struct android_app* app;

	ASensorManager* sensorManager;
	const ASensor* accelerometerSensor;
	ASensorEventQueue* sensorEventQueue;

	int animating;
	EGLDisplay display;
	EGLSurface surface;
	EGLContext context;
	int32_t width;
	int32_t height;
	struct saved_state state;
};

///////////////////

class System : public Singleton<System>
{
	friend class Singleton<System>;

private:
	System();

	Engine* m_engine;
	bool m_running;

	Scene* m_scene;

	inline unsigned int RunAndroid();
	inline unsigned int ShutdownAndroid();
	static void AHandleCmd(android_app* app, int32_t cmd);
	unsigned int Tick();
public:
	System(const System*);
	~System();

	unsigned int InitAndroid(android_app* app);
	unsigned int Initialize();
	unsigned int Shutdown();
	unsigned int Run();
	void Stop();

	bool GetRunning();
	Scene* GetCurrentScene();
	Engine* GetEngineData();
};

