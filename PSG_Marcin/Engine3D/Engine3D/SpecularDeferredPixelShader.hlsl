// globals
#define SPECULAR_INTENSITY_MULTIPLIER 3
#define ATTENUATION_MULTIPLIER 70.0f
#define MINIMUM_LENGTH_VALUE 0.00000000001f
#define LIGHT_MAX_COUNT 51

Texture2D colorTexture : register(t0);
Texture2D normalTexture : register(t1);
Texture2D worldPosTexture : register(t2);
SamplerState SamplerPoint : register(s0);

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
};

float4 SpecularDeferredPixelShader(PixelInput input) : SV_TARGET
{
	float4 colors;
	float4 normals;
	float4 worldPos;
	float3 r, v;
	float4 specular;
	float3 lightDir;
	float lightIntensity;
	float4 outputColor = float4(0, 0, 0, 0);
	float4 finalColor = float4(0, 0, 0, 0);
	float4 color;
	float lightCount = diffuseColors[0].w;

	float4 newPos;
	float att;
	float l;
	float4 newViewVector = float4(viewVector.x, viewVector.y, viewVector.z, 1.0f);

	// get color and normal from render textures
	colors = colorTexture.Sample(SamplerPoint, input.tex);
	normals = normalTexture.Sample(SamplerPoint, input.tex);
	worldPos = worldPosTexture.Sample(SamplerPoint, input.tex);

	for (int i = 0; i < lightCount; i++)
	{
		att = lightDirections[i].w;
		newPos = float4(lightDirections[i].x, lightDirections[i].y, lightDirections[i].z, 1.0f);

		lightDir = -normalize(worldPos - newPos);
		l = distance(worldPos, newPos);
		lightIntensity = saturate(dot(normals.xyz, lightDir));

		// compute for specular
		r = normalize(reflect(-lightDir, normals));
		v = normalize(newViewVector - worldPos);

		float dotProduct = saturate(dot(r, v));
		specular = SPECULAR_INTENSITY_MULTIPLIER*specularIntensity * specularColor * max(pow(dotProduct, glossiness), 0) * length(colors) * lightIntensity;

		color = saturate(lightIntensity*diffuseColors[i]);

		//outputColor *= saturate(ATTENUATION_MULTIPLIER*length(diffuseColors[i])*att / max(l*l, MINIMUM_LENGTH_VALUE));

		outputColor = saturate(color*colors + specular*colors.a*diffuseColors[i]);
		finalColor += outputColor*saturate(ATTENUATION_MULTIPLIER*length(diffuseColors[i])*att / max(l*l, MINIMUM_LENGTH_VALUE));
	}

	return saturate(finalColor + ambientColor*colors);
}