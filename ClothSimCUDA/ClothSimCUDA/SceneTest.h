#pragma once

/*
	This scene is used only for testing.
*/
#include "Scene.h"

class SceneTest :
	public Scene
{
public:
	SceneTest(string);
	SceneTest(const SceneTest*);
	~SceneTest();

	virtual unsigned int Initialize();
};

