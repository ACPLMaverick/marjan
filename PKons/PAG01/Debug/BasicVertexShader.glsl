#version 440 core

layout(location = 0) in vec3 vertexPosition_modelspace;
layout(location = 1) in vec3 vertexColor;
layout(location = 2) in vec2 vertexUV;
layout(location = 3) in vec3 vertexNormal;
uniform mat4 mvpMatrix;
uniform mat4 modelMatrix;

out vec3 modelPos;
out vec3 fragmentColor;
out vec2 UV;
out vec3 normal;
out vec4 lightDiff;

void main()
{
	vec4 v = vec4(vertexPosition_modelspace, 1.0f);
	modelPos = vec3(vec4(vertexPosition_modelspace, 1.0f) * modelMatrix);
	gl_Position = mvpMatrix * v;
	fragmentColor = vertexColor;
	UV = vertexUV;
	normal = vertexNormal;
	//normal = vec3(vec4(vertexNormal, 1.0f) * modelMatrix);
}