#version 400 core

#define POWER_CORRECTION 1.0f

in vec4 ProjPos;
in vec4 WorldPos;
in vec4 Vcol;
in vec3 Normal;
in vec2 UV;

out vec4 color;

// uniform sampler2D sampler;
// uniform mat4 mvpMatrix;
// uniform mat4 modelMatrix;
uniform vec4 EyeVector;
uniform vec4 LightDir;
uniform vec4 LightDiff;
uniform vec4 LightSpec;
uniform vec4 LightAmb;
// uniform vec4 highlight;
uniform float Gloss;

void main()
{
	// vec3 eyeP = normalize(modelPos - eyeVector.xyz);
	// vec4 vertexColor = vec4(fragmentColor.xyz, 1.0f);
	// vec4 tempColor = clamp(texture(sampler, UV)*vertexColor, 0.0f, 1.0f);
	// float lightPower = clamp(dot(-(lightDirection).xyz, normal), 0.0f, 1.0f);
	// vec3 diff = POWER_CORRECTION * lightPower * lightDiffuse.xyz * highlight.xyz;

	// vec3 r = normalize(reflect(-lightDirection.xyz, normal));
	// float spec = max(0.0f, dot(eyeP, r));
	// vec3 specFinal = POWER_CORRECTION * lightSpecular.xyz * lightPower * pow(spec, glossiness) * tempColor.a;

	// color = clamp(((diff + POWER_CORRECTION * lightAmbient.rgb)*tempColor.xyz + specFinal), 0.0f, 1.0f);
	color = Vcol;
}