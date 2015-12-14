#version 300 es

layout(location = 0) in vec4 Pos;					//current position
layout(location = 2) in vec4 Normal;	
layout(location = 3) in vec4 Neighbours;			//neighbours id
layout(location = 6) in vec4 NeighbourMultipliers;	// neighbour multipliers (i.e. do I have to take it into consideration)

uniform InPos
{
	vec4[16384] InPosBuffer;
};

out vec4 OutNormal;

void main()
{
	int mID = gl_VertexID;
	
	// calculate normals according to neighbours' positions

	vec3 normal = vec3(0.0f, 0.0f, 0.0f);
	vec3 mPos = vec3(Pos);
	for(int i = 0; i < 4; ++i)
	{
		int nID1 = int(roundEven(Neighbours[i]));
		int nID2 = int(roundEven(Neighbours[(i + 1) % 4]));

		vec3 diff1 = mPos - vec3(InPosBuffer[nID1]);
		vec3 diff2 = mPos - vec3(InPosBuffer[nID2]);

		normal = normal + (cross(diff1, diff2) * NeighbourMultipliers[i] * NeighbourMultipliers[(i + 1) % 4]);
	}
	normal = normalize(normal);
	normal.z = -normal.z;

	OutNormal = vec4(normal, 1.0f);
	gl_Position = Pos;
}