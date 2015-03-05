#version 330 core

layout(location = 0) in vec3 vertexPosition_modelspace;
uniform mat4 mvpMatrix;

void main()
{
	vec4 v = vec4(vertexPosition_modelspace, 1.0f);
	gl_Position = mvpMatrix * v;
}