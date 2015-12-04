#version 300 es
#extension GL_OES_standard_derivatives : enable 

in vec4 Vcol;
in vec3 Bar;

out vec4 color;

void main()
{
	/*
	vec3 d = fwidth(Bar);
    vec3 a3 = smoothstep(vec3(0.0), d*1.5, Bar);
    float edgeFactor = min(min(a3.x, a3.y), a3.z);
    edgeFactor = 1.0f - edgeFactor;
    */
    
    float edgeFactor = 0.0f;
    if(any(lessThan(Bar, vec3(0.04))))
    {
    	edgeFactor = 1.0f;
	}
	color = vec4(1.0f - Vcol.r, 1.0f - Vcol.g, 1.0f - Vcol.b, edgeFactor);
}