#pragma once

#include "stdafx.h"
#include "Texture.h"
#include "Color32.h"

class Material
{
protected:

#pragma region Protected

	Texture* _mapDiffuse;
	Texture* _mapNormal;
	Color32 _colorAmbient;
	Color32 _colorDiffuse;
	Color32 _colorSpecular;
	float _coeffGloss;

#pragma endregion

public:

#pragma region Functions Public

	Material(
		Texture* const diffuse = nullptr,
		Texture* const normal = nullptr,
		Color32 amb = Color32(0xFF000000),
		Color32 diff = Color32(0xFFFFFFFF),
		Color32 spec = Color32(0xFF000000),
		float gloss = 100.0f
	);
	~Material();

#pragma region Accessors

	const Texture* GetMapDiffuse() const { return _mapDiffuse; }
	const Texture* GetMapNormal() const { return _mapNormal; }
	const Color32* GetColorAmbient() const { return &_colorAmbient; }
	const Color32* GetColorDiffuse() const { return &_colorDiffuse; }
	const Color32* GetColorSpecular() const { return &_colorSpecular; }
	float GetCoefficentGloss() const { return _coeffGloss; }

	void SetMapDiffuse(Texture* const tex) { _mapDiffuse = tex; }
	void SetMapNormal(Texture* const tex) { _mapNormal = tex; }
	void SetColorAmbient(Color32 col) { _colorAmbient = col; }
	void SetColorDiffuse(Color32 col) { _colorDiffuse = col; }
	void SetColorSpecular(Color32 col) { _colorSpecular = col; }
	void SetCoefficentGloss(float g) { _coeffGloss = g; }

#pragma endregion

#pragma endregion

};
