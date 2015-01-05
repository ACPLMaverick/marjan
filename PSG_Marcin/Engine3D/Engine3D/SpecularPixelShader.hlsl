// globals
Texture2D shaderTexture;
SamplerState sampleType;

cbuffer LightBuffer
{
	float4 diffuseColor;
	float3 lightDirection;
	float padding;
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
};

float4 SpecularPixelShader(PixelInput input) : SV_TARGET
{
	float4 textureColor;
	float3 lightDir;
	float lightIntensity;
	float4 color;

	float3 r;
	float3 v;
	float4 specular;

	//float specularIntensity = 1.0f;
	//float glossiness = 100;
	//float4 specularColor = float4(1, 1, 1, 1);

	// sample pixel color from texture using sampler at this texture coordinate location
	textureColor = shaderTexture.Sample(sampleType, input.tex);

	lightDir = -lightDirection;
	lightIntensity = saturate(dot(input.normal, lightDir));

	// compute for specular
	r = normalize(lightDir - 2 * dot(lightDir, input.normal) * input.normal);
	v = viewVector;

	float dotProduct = dot(r, v);
	specular = specularIntensity * specularColor * max(pow(dotProduct, glossiness), 0) * length(textureColor) * lightIntensity;

	// combine with diffuse
	color = saturate(diffuseColor*lightIntensity + ambientColor);

	color = color*textureColor + specular*textureColor.a;

	return color;
}