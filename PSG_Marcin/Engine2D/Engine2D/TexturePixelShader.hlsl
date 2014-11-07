// globals
Texture2D shaderTexture;
SamplerState sampleType;

cbuffer TransparentBuffer
{
	float blendAmount;
};

// structs
struct PixelInput
{
	float4 position : SV_POSITION;
	float2 tex : TEXCOORD0;
};

float4 TexturePixelShader(PixelInput input) : SV_TARGET
{
	float4 textureColor;

	// sample pixel color from texture using sampler at this texture coordinate location
	textureColor = shaderTexture.Sample(sampleType, input.tex);
	textureColor.b = textureColor.r;
	textureColor.r = textureColor.g;
	textureColor.g = textureColor.b;
	//textureColor.a *= blendAmount;
	return textureColor;
}