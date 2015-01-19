// globals
Texture2D shaderTexture;
SamplerState sampleType;

cbuffer LightBuffer
{
	float4 diffuseColors[16];
	float4 lightDirections[16];
};

cbuffer AmbientLightBuffer
{
	float4 ambientColor;
};

// structs
struct PixelInput
{
	float4 position : SV_POSITION;
	float2 tex : TEXCOORD0;
	float3 normal : NORMAL;
};

float4 DiffusePixelShader(PixelInput input) : SV_TARGET
{
	float4 textureColor;
	float3 lightDir;
	float lightIntensity;
	float4 color;
	float4 finalColor = float4(0, 0, 0, 0);
	float lightCount = lightDirections[0].w;

	// sample pixel color from texture using sampler at this texture coordinate location
	textureColor = shaderTexture.Sample(sampleType, input.tex);

	for (int i = 0; i < lightCount; i++)
	{
		lightDir = -lightDirections[i];
		lightIntensity = saturate(dot(input.normal, lightDir));

		// combine with diffuse
		color = saturate(diffuseColors[i] * lightIntensity + ambientColor);

		finalColor = saturate(finalColor + color*textureColor);
	}

	return finalColor;
}