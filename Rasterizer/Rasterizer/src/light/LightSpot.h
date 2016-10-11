#pragma once

#include "LightDirectional.h"

namespace light
{
	class LightSpot : public LightDirectional
	{
	protected:

#pragma region Protected

		float _attenuationConstant = 0.0f;
		float _attenuationLinear = 1.0f;
		float _attenuationQuadratic = 1.0f;
		float _umbraAngleRad = (float)M_PI_2;
		float _penumbraAngleRad = (float)M_PI_4;
		float _falloffFactor = 1.0f;

#pragma endregion

	public:

#pragma region Functions Public

		LightSpot();
		LightSpot(
			const Color32* col,
			const math::Float3* dir,
			float attC,
			float attL,
			float attQ,
			float umbra,
			float penumbra,
			float falloff
		);
		~LightSpot();

#pragma region Accessors

		float GetAttenuationConstant() const { return _attenuationConstant; }
		float GetAttenuationLinear() const { return _attenuationLinear; }
		float GetAttenuationQuadratic() const { return _attenuationQuadratic; }
		float GetFalloffFactor() const { return _falloffFactor; }
		float GetUmbraAngle() const;
		float GetPenumbraAngle() const;

		void SetAttenuationConstant(float ac) { _attenuationConstant = ac; }
		void SetAttenuationLinear(float al) { _attenuationLinear = al; }
		void SetAttenuationQuadratic(float aq) { _attenuationQuadratic = aq; }
		void SetFalloffFactor(float ff) { _falloffFactor = ff; }
		void SetUmbraAngle(float ua);
		void SetPenumbraAngle(float pa);

#pragma endregion

#pragma endregion
	};
}