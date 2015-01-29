// globals
cbuffer MatrixBuffer
{
	matrix worldMatrix;
	matrix viewMatrix;
	matrix projectionMatrix;
};

// structures
struct VertexInput
{
	float4 position : POSITION;
	float2 tex : TEXCOORD0;
	float3 normal : NORMAL;
};

struct PixelInput
{
	float4 position : SV_POSITION;
	float2 tex : TEXCOORD0;
	float3 normal : NORMAL;
	float4 worldPos : POSITION;
	matrix wm : TEXCOORD2;
};
//////////////////////
PixelInput SpecularVertexShader(VertexInput input)
{
	PixelInput output;

	input.position.w = 1.0f;

	output.position = mul(input.position, worldMatrix);
	output.worldPos = mul(input.position, worldMatrix);
	output.position = mul(output.position, viewMatrix);
	output.position = mul(output.position, projectionMatrix);

	// store texture coordinates for pixel shader!
	output.tex = input.tex;

	// calculate normal vector against world matrix
	output.normal = mul(input.normal, (float3x3)worldMatrix);

	//normalize vector
	output.normal = normalize(output.normal);

	output.wm = worldMatrix;

	return output;
}