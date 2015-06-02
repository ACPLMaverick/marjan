#include "TextureManager.h"

TextureManager* TextureManager::instance;

TextureManager::TextureManager()
{
	currentLoaded = -1;
	currentLoading = -1;
	for (int i = 0; i < FIXED_MIPMAP_COUNT; ++i)
	{
		mipmaps[i] = nullptr;
	}
}


TextureManager::~TextureManager()
{
	for (int i = 0; i < FIXED_MIPMAP_COUNT; ++i)
	{
		if (mipmaps[i] != nullptr)
		{
			mipmaps[i]->Shutdown();
			delete mipmaps[i];
		}
	}
}

bool TextureManager::LoadAsync(unsigned int which)
{
	if (which >= FIXED_MIPMAP_COUNT || instance->mipmaps[which] != nullptr)
	{
		return false;
	}
	instance->currentLoading = which;

	Bitmap* nbm = new Bitmap(instance->currentLoading);
	string filePath = FIXED_MIPMAP_PATH + to_string(instance->currentLoading) + FIXED_MIPMAP_EXT;
	nbm->Initialize(&filePath);
	instance->mipmaps[instance->currentLoading] = nbm;

	return true;
}

void TextureManager::TextureLoadedCallback(Bitmap* texture)
{
	printf("TextureManager: Loaded mip %d\n", texture->GetMipID());
	instance->currentLoaded = instance->currentLoading;

	// deleting other mipmaps
	for (int i = 0; i < FIXED_MIPMAP_COUNT; ++i)
	{
		if (instance->mipmaps[i] != nullptr && i != instance->currentLoaded)
		{
			instance->mipmaps[i]->Shutdown();
			delete instance->mipmaps[i];
			instance->mipmaps[i] = nullptr;
		}
	}

	Graphics::Get()->SwapTexture(texture);
}

TextureManager* TextureManager::Get()
{
	if (instance == nullptr)
	{
		instance = new TextureManager();
	}
	
	return instance;
}

void TextureManager::Destroy()
{
	if (instance != nullptr)
	{
		delete instance;
	}
}

void TextureManager::CheckForLoadedBitmaps()
{
	for (int i = 0; i < FIXED_MIPMAP_COUNT; ++i)
	{
		if (instance->mipmaps[i] != nullptr && !((Bitmap*)(instance->mipmaps[i]))->Loaded)
		{
			((Bitmap*)(instance->mipmaps[i]))->CheckIfLoaded();
		}
	}
}
