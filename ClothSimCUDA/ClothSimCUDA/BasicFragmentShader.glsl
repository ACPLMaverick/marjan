#version 400 core

#define POWER_CORRECTION 1.0f

in vec3 fragmentColor;
in vec2 UV;
in vec3 normal;
in vec3 modelPos;

out vec3 color;

uniform sampler2D sampler;
uniform mat4 mvpMatrix;
uniform mat4 modelMatrix;
uniform vec4 eyeVector;
uniform vec4 lightDirection;
uniform vec4 lightDiffuse;
uniform vec4 lightSpecular;
uniform vec4 lightAmbient;
uniform vec4 highlight;
uniform float glossiness;

void main()
{
	vec3 eyeP = normalize(modelPos - eyeVector.xyz);
	vec4 vertexColor = vec4(fragmentColor.xyz, 1.0f);
	vec4 tempColor = clamp(texture(sampler, UV)*vertexColor, 0.0f, 1.0f);
	float lightPower = clamp(dot(-(lightDirection).xyz, normal), 0.0f, 1.0f);
	vec3 diff = POWER_CORRECTION * lightPower * lightDiffuse.xyz * highlight.xyz;

	vec3 r = normalize(reflect(-lightDirection.xyz, normal));
	float spec = max(0.0f, dot(eyeP, r));
	vec3 specFinal = POWER_CORRECTION * lightSpecular.xyz * lightPower * pow(spec, glossiness) * tempColor.a;

	color = clamp(((diff + POWER_CORRECTION * lightAmbient.rgb)*tempColor.xyz + specFinal), 0.0f, 1.0f);
}