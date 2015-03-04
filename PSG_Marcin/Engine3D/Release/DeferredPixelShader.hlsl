Texture2D shaderTexture : register(t0);
SamplerState SampleTypeWrap : register(s0);

struct PixelInput
{
	float4 position : SV_POSITION;
	float2 tex : TEXCOORD0;
	float3 normal : NORMAL;
	float4 worldPos : TEXCOORD1;
};

struct PixelOutput
{
	float4 color : SV_Target0;
	float4 normal : SV_Target1;
	float4 worldPos : SV_Target2;
};

PixelOutput DeferredPixelShader(PixelInput input) : SV_TARGET
{
	PixelOutput output;

	output.color = shaderTexture.Sample(SampleTypeWrap, input.tex);

	output.normal = float4(normalize(input.normal), 1.0f);

	output.worldPos = input.worldPos;

	return output;
}