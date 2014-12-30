// globals
Texture2D shaderTexture;
SamplerState sampleType;

cbuffer LightBuffer
{
	float4 diffuseColor;
	float3 lightDirection;
	float padding;
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

	// sample pixel color from texture using sampler at this texture coordinate location
	textureColor = shaderTexture.Sample(sampleType, input.tex);

	lightDir = -lightDirection;
	lightIntensity = saturate(dot(input.normal, lightDir));

	// combine with diffuse
	color = saturate(diffuseColor*lightIntensity);

	color = color*textureColor;

	return color;
}