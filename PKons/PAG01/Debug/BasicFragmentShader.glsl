#version 440 core

#define POWER_CORRECTION 10.0f

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
uniform float glossiness;

void main()
{
	//vec3 eyeP = normalize(modelPos - eyeVector.xyz);
	vec3 tempColor = clamp(texture(sampler, UV).rgb*fragmentColor, 0.0f, 1.0f);
	float lightPower = clamp(dot(-(lightDirection).xyz, normal), 0.0f, 1.0f);
	vec3 diff = POWER_CORRECTION * lightPower * lightDiffuse.xyz;

	vec3 r = normalize(reflect(-lightDirection.xyz, normal));
	float spec = max(0.0f, dot(normalize(eyeVector.xyz), r));
	vec3 specFinal = POWER_CORRECTION * lightSpecular.xyz * lightPower * pow(spec, glossiness);

	color = clamp((diff + POWER_CORRECTION * lightAmbient.rgb)*tempColor + specFinal, 0.0f, 1.0f);
}