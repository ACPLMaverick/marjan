#pragma once

#include <Windows.h>
#include <GL\glew.h>
#include <GLFW\glfw3.h>
#include <glm\glm\glm.hpp>

#include "TextureManager.h"
#include "Texture.h"

#define FIXED_CHANNELS 4

class TextureManager;

class Bitmap :
	public Texture
{
private:
	static Bitmap* currentlyLoading;

	unsigned int mipID;
	HANDLE hFile;
	OVERLAPPED asyncHandler;
	char* data;
	unsigned int dataSize;

	DWORD LoadAsync(const string* filePath);
	static void LoadedCallback();
	DWORD CloseReadOperation();
public:
	bool Loaded;

	Bitmap(unsigned int mid);
	~Bitmap();

	bool Initialize(const string* filePath);
	void Shutdown() override;
	unsigned int GetMipID();
	bool CheckIfLoaded();
};

