#version 300 es

layout(location = 0) in vec4 Pos;					//current position
layout(location = 1) in vec4 PosLast;			//previous position

uniform InPos
{
	vec4[16384] InPosBuffer;
};
uniform InPosLast
{
	vec4[16384] InPosLastBuffer;
};

out vec4 OutPos;
out vec4 OutPosLast;

void main()
{
	int id = gl_VertexID;
	vec4 dupa = InPosBuffer[id];
	vec4 pos = vec4(dupa.x, dupa.y - 0.01f, dupa.z, dupa.w);
	OutPos = pos;
	OutPosLast = Pos;
	gl_Position = pos;
}