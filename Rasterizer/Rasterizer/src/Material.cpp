#include "Material.h"



Material::Material
(
	Texture* const diffuse,
	Texture* const normal,
	Color32 amb,
	Color32 diff,
	Color32 spec,
	float gloss
) :
	_colorAmbient(amb),
	_colorDiffuse(diff),
	_colorSpecular(spec),
	_coeffGloss(gloss),
	_mapDiffuse(diffuse),
	_mapNormal(normal)
{
}


Material::~Material()
{
	if (_mapDiffuse != nullptr)
	{
		delete _mapDiffuse;
	}
	if (_mapNormal != nullptr)
	{
		delete _mapNormal;
	}
}
