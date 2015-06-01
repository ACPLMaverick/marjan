#pragma once

#include <GL\glew.h>
#include <GLFW\glfw3.h>
#include <glm\glm\glm.hpp>
#include <glm\glm\gtx\transform.hpp>
#include <iostream>
#include <string>

#include "Texture.h"
#include "Bitmap.h"
#include "Graphics.h"

#define FIXED_MIPMAP_COUNT 6
#define FIXED_MIPMAP_PATH "mipmaps_t\\mip"
#define FIXED_MIPMAP_EXT ".bmp"

using namespace std;

class Bitmap;

class TextureManager
{
private:
	static TextureManager* instance;

	Texture* mipmaps[FIXED_MIPMAP_COUNT];
	unsigned int currentLoaded;
	unsigned int currentLoading;

public:
	TextureManager();
	~TextureManager();

	static bool LoadAsync(unsigned int which);
	static TextureManager* Get();
	static void Destroy();
	static void TextureLoadedCallback(Bitmap* texture);
	static void CheckForLoadedBitmaps();
};

