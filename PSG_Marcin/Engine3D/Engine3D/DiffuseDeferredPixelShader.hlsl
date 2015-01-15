// globals
Texture2D colorTexture : register(t0);
Texture2D normalTexture : register(t1);
SamplerState SamplerPoint : register(s0);

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
};

float4 DiffuseDeferredPixelShader(PixelInput input) : SV_TARGET
{
	float4 colors;
	float4 normals;
	float3 lightDir;
	float lightIntensity;
	float4 outputColor = float4(0,0,0,0);
	float lightCount = lightDirections[0].w;

	// get color and normal from render textures
	colors = colorTexture.Sample(SamplerPoint, input.tex);
	normals = normalTexture.Sample(SamplerPoint, input.tex);

	for (int i = 0; i < lightCount; i++)
	{
		lightDir = -lightDirections[i].xyz;
		lightIntensity = saturate(dot(normals.xyz, lightDir));
		outputColor = saturate(outputColor + saturate(colors*lightIntensity*diffuseColors[i]));
	}
	outputColor = saturate(outputColor + ambientColor);

	return outputColor;
}