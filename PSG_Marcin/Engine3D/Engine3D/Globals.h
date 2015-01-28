#pragma once

#define WIN32_LEAN_AND_MEAN

#define SCREEN_WIDTH 800
#define SCREEN_HEIGHT 600

#define VK_LETTER_A 0x41
#define VK_LETTER_W 0x57
#define VK_LETTER_S 0x53
#define VK_LETTER_D 0x44
#define VK_LETTER_E 0x45

#define LIGHT_MAX_COUNT 100
#define OBJECTS_COUNT 5

static int DEFERRED = 0;
static const int TESTMODE = 1;

static const char* SCENE_PATH_DEFERRED = "./Scenes/TestSceneDeferred.sc";
static const char* SCENE_PATH = "./Scenes/TestScene.sc";