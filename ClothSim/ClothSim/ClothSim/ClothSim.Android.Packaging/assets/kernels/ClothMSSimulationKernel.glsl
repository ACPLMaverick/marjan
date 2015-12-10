#version 300 es

layout(location = 0) in vec3 Pos;					//current position
layout(location = 1) in vec3 PosLast;			//previous position

out vec3 OutPos;
out vec3 OutPosLast;

void main()
{
	int id = gl_VertexID;
	vec3 pos = vec3(Pos.x, Pos.y - 0.001f, Pos.z);
	OutPos = pos;
	OutPosLast = Pos;
	gl_Position = vec4(pos, 1.0f);
}