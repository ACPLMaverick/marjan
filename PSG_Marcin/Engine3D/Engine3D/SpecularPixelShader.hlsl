// globals
Texture2D shaderTexture;
SamplerState sampleType;

cbuffer LightBuffer
{
	float4 diffuseColors[100];
	float4 lightDirections[100];
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
	matrix worldMatrix : TEXCOORD1;
};

float4 SpecularPixelShader(PixelInput input) : SV_TARGET
{
	float4 textureColor;
	float3 lightDir;
	float lightIntensity;
	float4 color;
	float4 finalColor = float4(0, 0, 0, 0);
	float lightCount = lightDirections[0].w;

	float3 r;
	float3 v;
	float4 specular;

	// sample pixel color from texture using sampler at this texture coordinate location
	textureColor = shaderTexture.Sample(sampleType, input.tex);

	for (int i = 0; i < lightCount; i++)
	{
		lightDir = -lightDirections[i];
		lightIntensity = saturate(dot(input.normal, lightDir));

		// compute for specular
		r = normalize(lightDir - 2 * dot(lightDir, input.normal) * input.normal);
		v = normalize(mul(viewVector, input.worldMatrix) - mul(input.position, input.worldMatrix));

		float dotProduct = saturate(dot(r, v));
		specular = specularIntensity * specularColor * max(pow(dotProduct, glossiness), 0) * length(textureColor) * lightIntensity;

		// combine with diffuse
		color = saturate(diffuseColors[i]*lightIntensity + ambientColor);

		finalColor = saturate(finalColor + color*textureColor + specular*textureColor.a*diffuseColors[i]);
	}

	return finalColor;
}