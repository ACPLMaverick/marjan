// globals
Texture2D colorTexture : register(t0);
Texture2D normalTexture : register(t1);
SamplerState SamplerPoint : register(s0);

cbuffer LightBuffer
{
	float4 diffuseColors[1000];
	float4 lightDirections[1000];
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
	float3 r, v;
	float4 specular;
	float3 lightDir;
	float lightIntensity;
	float4 outputColor = float4(0, 0, 0, 0);
	float4 color;
	float lightCount = lightDirections[0].w;

	// get color and normal from render textures
	colors = colorTexture.Sample(SamplerPoint, input.tex);
	normals = normalTexture.Sample(SamplerPoint, input.tex);

	for (int i = 0; i < lightCount; i++)
	{
		lightDir = -lightDirections[i].xyz;
		lightIntensity = saturate(dot(normals.xyz, lightDir));

		// compute for specular
		r = normalize(lightDir - 2 * dot(-lightDir, normals.xyz) * normals.xyz);
		v = viewVector;

		float dotProduct = saturate(dot(r, v));
		specular = specularIntensity * specularColor * max(pow(dotProduct, glossiness), 0) * length(colors) * lightIntensity;

		color = saturate(lightIntensity*diffuseColors[i] + ambientColor);
		outputColor = saturate(outputColor + color*colors + specular*colors.a*diffuseColors[i]);
	}

	return outputColor;
}