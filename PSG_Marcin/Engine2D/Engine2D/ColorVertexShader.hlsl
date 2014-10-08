

// GLOBALS
cbuffer MatrixBuffer
{
	matrix worldMatrix;
	matrix viewMatrix;
	matrix projectionMatrix;
};

// Structures
struct VertexInput
{
	float4 position : POSITION;
	float4 color : COLOR;
};

struct PixelInput
{
	float4 position : SV_POSITION;
	float4 color : COLOR;
};
[numthreads(1, 1, 1)]
// vertex shader
PixelInput main( VertexInput input /*uint3 DTid : SV_DispatchThreadID*/ ) : SV_TARGET
{
	PixelInput output;

	// change the position vector to be 4 units for proper matrix calculations
	input.position.w = 1.0f;

	// calculate position of vertex against the world, view and
	// projection matrices
	output.position = mul(input.position, worldMatrix);
	output.position = mul(output.position, viewMatrix);
	output.position = mul(output.position, projectionMatrix);

	output.color = input.color;

	return output;
}