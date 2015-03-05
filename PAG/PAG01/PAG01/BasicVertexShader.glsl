#version 330 core

layout(location = 0) in vec3 vertexPosition_modelspace;
layout(location = 1) in vec3 vertexColor;
uniform mat4 mvpMatrix;

out vec3 fragmentColor;

void main()
{
	vec4 v = vec4(vertexPosition_modelspace, 1.0f);
	gl_Position = mvpMatrix * v;
	fragmentColor = vertexColor;
}