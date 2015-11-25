#include "System.h"

System::System()
{

}

System::System(const System*)
{

}


System::~System()
{

}



unsigned int System::Initialize(android_app* app)
{
	unsigned int err = CS_ERR_NONE;

	m_running = true;

	// initialize android stuff common to all users
	err = InitAndroid(app);
	if (err != CS_ERR_NONE) return err;

	// initializing main singletons

	err = Renderer::GetInstance()->Initialize();
	if (err != CS_ERR_NONE) return err;

	err = ResourceManager::GetInstance()->Initialize();
	if (err != CS_ERR_NONE) return err;

	err = InputManager::GetInstance()->Initialize();
	if (err != CS_ERR_NONE) return err;

	err = PhysicsManager::GetInstance()->Initialize();
	if (err != CS_ERR_NONE) return err;

	InputHandler::GetInstance();

	Timer::GetInstance()->Initialize();

	// initializing scene and objects
	m_scene = new SceneTest("SceneTest");
	err = m_scene->Initialize();
	if (err != CS_ERR_NONE) return err;

	// initializing gui

	// initializing CUDA

#ifdef _DEBUG
	printf("\nSystem.Initialize has completed successfully.\n");
#endif
	return err;
}

unsigned int System::Shutdown()
{
	unsigned int err = CS_ERR_NONE;

	m_running = false;

	// shutting down scene and objects
	err = m_scene->Shutdown();
	if (err != CS_ERR_NONE) return err;

	// shutting down main singletons

	err = Renderer::GetInstance()->Shutdown();
	if (err != CS_ERR_NONE) return err;

	err = InputManager::GetInstance()->Shutdown();
	if (err != CS_ERR_NONE) return err;

	err = PhysicsManager::GetInstance()->Shutdown();
	if (err != CS_ERR_NONE) return err;

	Renderer::DestroyInstance();
	InputManager::DestroyInstance();
	InputHandler::DestroyInstance();
	PhysicsManager::DestroyInstance();

	// shutting down gui

	// shutting down CUDA

#ifdef _DEBUG
	printf("\nSystem.Shutdown has completed successfully.\n");
#endif
	return err;
}

// main loop runs here
unsigned int System::Run()
{
	unsigned int err = CS_ERR_NONE;

	while (m_running)
	{
		// update timer
		Timer::GetInstance()->Run();

		// update android-related stuff, mainly events
		RunAndroid();

		// update input
		InputManager::GetInstance()->Run();

		// update gui

		// update scene
		m_scene->Update();

		// compute collisions
		PhysicsManager::GetInstance()->Run();

		// draw one frame
		Renderer::GetInstance()->Run();
	}

	return err;
}

void System::Stop()
{
	m_running = false;
}


/**
* Initialize android necessary stuff
*/
unsigned int System::InitAndroid(android_app* app)
{
	m_engine = new Engine();
	app->userData = m_engine;
	app->onAppCmd = System::AHandleCmd;
	app->onInputEvent = InputManager::AHandleInput;
	m_engine->app = app;

	// Prepare to monitor accelerometer
	m_engine->sensorManager = ASensorManager_getInstance();
	m_engine->accelerometerSensor = ASensorManager_getDefaultSensor(m_engine->sensorManager,
		ASENSOR_TYPE_ACCELEROMETER);
	m_engine->sensorEventQueue = ASensorManager_createEventQueue(m_engine->sensorManager,
		m_engine->app->looper, LOOPER_ID_USER, NULL, NULL);

	if (m_engine->app->savedState != NULL) {
		// We are starting with a previous saved state; restore from it.
		m_engine->state = *(struct saved_state*)app->savedState;
	}

	m_engine->animating = 1;

	// loop waiting for recieving context from android

	while (true)
	{
		int ident;
		int events;
		android_poll_source* source;

		// loop until all events are read, then continue
		// this is necessary to recieve ANativeActivity pointer
		while (
			ident = ALooper_pollAll(m_engine->animating ? 0 : -1, NULL, &events, (void**)&source) >= 0
			)
		{
			// process this event
			if (source != NULL)
			{
				source->process(m_engine->app, source);
			}

			// originally sensor data processing was here

			// check if we are exiting
			if (m_engine->app->destroyRequested != 0)
			{
				return CS_ANDROID_ERROR;
			}
		}
	}

	return CS_ERR_NONE;
}

/**
* Destroy android stuff
*/
void System::ShutdownAndroid()
{
	delete m_engine;
}

/**
* Run android stuff, mainly events
*/
void System::RunAndroid()
{
	// Read all pending events.
	int ident;
	int events;
	struct android_poll_source* source;

	// If not animating, we will block forever waiting for events.
	// If animating, we loop until all events are read, then continue
	// to draw the next frame of animation.
	while ((ident = ALooper_pollAll(m_engine->animating ? 0 : -1, NULL, &events,
		(void**)&source)) >= 0) {

		// Process this event.
		if (source != NULL) {
			source->process(m_engine->app, source);
		}

		// If a sensor has data, process it now.
		if (ident == LOOPER_ID_USER) {
			// Checking for accelrometer data is commented out.

			if (m_engine->accelerometerSensor != NULL) {
				ASensorEvent event;
				while (ASensorEventQueue_getEvents(m_engine->sensorEventQueue,
					&event, 1) > 0) {
					/*
					LOGI("accelerometer: x=%f y=%f z=%f",
					event.acceleration.x, event.acceleration.y,
					event.acceleration.z);*/
				}
			}

		}

		// Check if we are exiting.
		if (m_engine->app->destroyRequested != 0) {
			Renderer::GetInstance()->Shutdown();
			return;
		}
	}
}

/**
* Process the next main command.
*/
void System::AHandleCmd(struct android_app* app, int32_t cmd) {
	struct Engine* engine = (struct Engine*)app->userData;
	switch (cmd) {
	case APP_CMD_SAVE_STATE:
		// The system has asked us to save our current state.  Do so.
		engine->app->savedState = malloc(sizeof(struct saved_state));
		*((struct saved_state*)engine->app->savedState) = engine->state;
		engine->app->savedStateSize = sizeof(struct saved_state);
		break;
	case APP_CMD_INIT_WINDOW:
		// The window is being shown, get it ready.
		if (engine->app->window != NULL) {
			Renderer::GetInstance()->Initialize();
			Renderer::GetInstance()->Run();
		}
		break;
	case APP_CMD_TERM_WINDOW:
		// The window is being hidden or closed, clean it up.
		Renderer::GetInstance()->Shutdown();
		break;
	case APP_CMD_GAINED_FOCUS:
		// When our app gains focus, we start monitoring the accelerometer.
		if (engine->accelerometerSensor != NULL) {
			ASensorEventQueue_enableSensor(engine->sensorEventQueue,
				engine->accelerometerSensor);
			// We'd like to get 60 events per second (in us).
			ASensorEventQueue_setEventRate(engine->sensorEventQueue,
				engine->accelerometerSensor, (1000L / 60) * 1000);
		}
		break;
	case APP_CMD_LOST_FOCUS:
		// When our app loses focus, we stop monitoring the accelerometer.
		// This is to avoid consuming battery while not being used.
		if (engine->accelerometerSensor != NULL) {
			ASensorEventQueue_disableSensor(engine->sensorEventQueue,
				engine->accelerometerSensor);
		}
		// Also stop animating.
		engine->animating = 0;
		Renderer::GetInstance()->Run();
		break;
	}
}


Scene* System::GetCurrentScene()
{
	return m_scene;
}

Engine* System::GetEngineData()
{
	return m_engine;
}
