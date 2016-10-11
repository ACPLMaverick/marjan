#include "LightSpot.h"
#include "../Math.h"

namespace light
{
	LightSpot::LightSpot() :
		LightDirectional()
	{
	}

	LightSpot::LightSpot(const Color32 * col, const math::Float3* dir, float attC, float attL, float attQ, float umbra, float penumbra, float falloff) :
		LightDirectional(col, dir),
		_attenuationConstant(attC),
		_attenuationLinear(attL),
		_attenuationQuadratic(attQ),
		_umbraAngleRad(DegToRad(umbra)),
		_penumbraAngleRad(DegToRad(penumbra)),
		_falloffFactor(falloff)
	{
	}

	LightSpot::~LightSpot()
	{
	}

	float LightSpot::GetUmbraAngle() const
	{
		return RadToDeg(_umbraAngleRad);
	}

	float LightSpot::GetPenumbraAngle() const
	{
		return RadToDeg(_penumbraAngleRad);
	}
	void LightSpot::SetUmbraAngle(float ua)
	{
		_umbraAngleRad = DegToRad(ua);
	}

	void LightSpot::SetPenumbraAngle(float pa)
	{
		_penumbraAngleRad = DegToRad(pa);
	}
}