#version 300 es

layout(location = 0) in vec3 Pos;					//current position
layout(location = 1) in vec3 PosLast;			//previous position

struct InBuffer
{
	vec3 buf;
	//half padding;
};

uniform InPos
{
	InBuffer[16384] InPosBuffer;
};
uniform InPosLast
{
	InBuffer[16384] InPosLastBuffer;
};

out vec3 OutPos;
out vec3 OutPosLast;

void main()
{
	int id = gl_VertexID;
	vec3 dupa = InPosBuffer[1].buf;
	vec3 pos = vec3(dupa.x, dupa.y, dupa.z);
	OutPos = pos;
	OutPosLast = Pos;
	gl_Position = vec4(pos, 1.0f);
}