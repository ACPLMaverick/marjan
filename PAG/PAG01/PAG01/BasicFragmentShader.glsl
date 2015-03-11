#version 330 core

in vec3 fragmentColor;
in vec2 UV;
in vec3 normal;

out vec3 color;

uniform sampler2D sampler;

uniform vec4 eyeVector;
uniform vec4 lightDirection;
uniform vec4 lightDiffuse;
uniform vec4 lightSpecular;
uniform vec4 lightAmbient;
uniform float glossiness;

void main()
{
	vec3 tempColor = clamp(texture(sampler, UV).rgb*fragmentColor, 0.0f, 1.0f);
	float lightPower = dot(-(lightDirection).xyz, normal);
	vec3 diff = lightPower * lightDiffuse.xyz;

	vec3 r = reflect(-lightDirection.xyz, normal);
	float spec = clamp(dot(eyeVector.xyz, r), 0.0f, 1.0f);
	vec3 specFinal = lightSpecular.xyz * lightPower * pow(spec, glossiness);

	color = clamp((diff + lightAmbient.rgb)*tempColor/* + specFinal*/, 0.0f, 1.0f);
}