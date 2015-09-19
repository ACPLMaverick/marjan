#version 400 core

in vec4 Vcol;
in vec2 UV;

out vec4 color;

uniform sampler2D sampler;

void main()
{
	vec4 finalColor = Vcol * texture(sampler, UV);
	color = finalColor;
}