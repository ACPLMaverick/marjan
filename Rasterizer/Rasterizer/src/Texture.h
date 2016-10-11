#pragma once

#include "stdafx.h"
#include "Color32.h"
#include "Float2.h"

#define TEXTURE_PATH "./textures/"
#define TEXTURE_EXTENSION ".tga"

class Texture
{
public:

#pragma region Enums Public

	enum WrapMode
	{
		WRAP,
		CLAMP
	};

	enum FilterMode
	{
		NEAREST,
		LINEAR
	};

#pragma endregion

protected:

#pragma region Protected

	Color32* _data = nullptr;
	WrapMode _wm = WrapMode::WRAP;
	FilterMode _fm = FilterMode::LINEAR;
	uint16_t _width = 0;

#pragma endregion

#pragma region Functions Protected

	inline void LoadFromFile(const std::string* name);

#pragma endregion

public:

#pragma region Functions Public

	Texture();
	Texture(Color32 col, WrapMode wm = WrapMode::WRAP, FilterMode fm = FilterMode::LINEAR);
	Texture(const std::string* name, WrapMode wm = WrapMode::WRAP, FilterMode fm = FilterMode::LINEAR);
	~Texture();

	Color32 GetColor(const math::Float2* uv) const;

#pragma region Accessors

	uint16_t GetWidth() const { return _width; }

#pragma endregion

#pragma endregion
};

