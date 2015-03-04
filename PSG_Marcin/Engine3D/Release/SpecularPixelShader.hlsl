// globals
#define SPECULAR_INTENSITY_MULTIPLIER 3
#define ATTENUATION_MULTIPLIER 70.0f
#define MINIMUM_LENGTH_VALUE 0.00000000001f
#define LIGHT_MAX_COUNT 51

Texture2D shaderTexture;
SamplerState sampleType;

cbuffer LightBuffer
{
	float4 diffuseColors[LIGHT_MAX_COUNT];
	float4 lightDirections[LIGHT_MAX_COUNT];
};

cbuffer AmbientLightBuffer
{
	float4 ambientColor;
};

cbuffer SpecularBuffer
{
	float3 viewVector;
	float specularIntensity;
	float4 specularColor;
	float glossiness;
	float3 padding02;
	float4 padding01;
};

// structs
struct PixelInput
{
	float4 position : SV_POSITION;
	float2 tex : TEXCOORD0;
	float3 normal : NORMAL;
	float4 worldPos : POSITION;
	matrix wm : TEXCOORD2;
};

float4 SpecularPixelShader(PixelInput input) : SV_TARGET
{
	float4 newPos;
	float att;
	float l;
	float4 textureColor;
	float3 lightDir;
	float lightIntensity;
	float4 color;
	float4 finalColor = float4(0, 0, 0, 0);
	float lightCount = diffuseColors[0].w;

	float4 newViewVector = float4(viewVector.x, viewVector.y, viewVector.z, 1.0f);

	float3 r;
	float3 v;
	float4 specular;

	float4 finalFinalColor = float4(0.0f, 0.0f, 0.0f, 0.0f);

	// sample pixel color from texture using sampler at this texture coordinate location
	textureColor = shaderTexture.Sample(sampleType, input.tex);

	for (int i = 0; i < lightCount; i++)
	{
		// calculate light direction at given pixel
		att = lightDirections[i].w;
		newPos = float4(lightDirections[i].x, lightDirections[i].y, lightDirections[i].z, 1.0f);
		lightDir = -normalize(input.worldPos - newPos);
		l = distance(input.worldPos, newPos);
		lightIntensity = saturate(dot(input.normal, lightDir));

		r = normalize(reflect(-lightDir, input.normal));
		v = normalize(newViewVector.xyz - input.worldPos.xyz);

		float dotProduct = max((dot(r, v)), 0.0f);
		specular = SPECULAR_INTENSITY_MULTIPLIER*specularIntensity * specularColor * max(pow(dotProduct, glossiness), 0) * length(textureColor) * lightIntensity;
		specular *= diffuseColors[i];

		// combine with diffuse
		color = saturate(diffuseColors[i]*lightIntensity);

		//attenuation
		//finalColor *= saturate(ATTENUATION_MULTIPLIER*length(diffuseColors[i])*att / max(l*l, MINIMUM_LENGTH_VALUE));

		finalColor = saturate(color*textureColor + specular*textureColor.a*diffuseColors[i]);
		finalFinalColor += finalColor*saturate(ATTENUATION_MULTIPLIER*length(diffuseColors[i])*att / max(l*l, MINIMUM_LENGTH_VALUE));
	}

	return saturate(finalFinalColor + ambientColor*textureColor);
}