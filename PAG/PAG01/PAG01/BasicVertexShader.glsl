#version 330 core

layout(location = 0) in vec3 vertexPosition_modelspace;
layout(location = 1) in vec3 vertexColor;
layout(location = 2) in vec2 vertexUV;
layout(location = 3) in vec3 vertexNormal;
uniform mat4 mvpMatrix;

out vec3 fragmentColor;
out vec2 UV;
out vec3 normal;
out vec4 lightDiff;

void main()
{
	vec4 v = vec4(vertexPosition_modelspace, 1.0f);
	gl_Position = mvpMatrix * v;
	fragmentColor = vertexColor;
	UV = vertexUV;
	normal = vertexNormal;
}