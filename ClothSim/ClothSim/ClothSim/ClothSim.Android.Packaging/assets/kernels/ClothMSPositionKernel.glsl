#version 300 es

layout(location = 0) in vec4 Pos;					//current position
layout(location = 1) in vec4 PosLast;				//previous position
layout(location = 2) in vec4 Normal;	
layout(location = 3) in vec4 Neighbours;			//neighbours id
layout(location = 4) in vec4 NeighbourMultipliers;	// neighbour multipliers (i.e. do I have to take it into consideration)
layout(location = 5) in vec4 SpringLengths;
layout(location = 6) in float Elasticity;
layout(location = 7) in float Mass;
layout(location = 8) in float ElDampCoeff;
layout(location = 9) in float AirDampCoeff;
layout(location = 10) in float LockMultiplier;
layout(location = 11) in float ColliderMultiplier;

uniform InPos
{
	vec4[16384] InPosBuffer;
};
uniform InPosLast
{
	vec4[16384] InPosLastBuffer;
};

uniform int VertexCount;
uniform int EdgesWidthAll;
uniform int EdgesLengthAll;
uniform float DeltaTime;
uniform float Gravity;

out vec4 OutPos;
out vec4 OutPosLast;

void main()
{
	int mID = gl_VertexID;
	vec3 mPos = vec3(Pos);
	vec3 mPosLast = vec3(PosLast);
	vec3 mVel = (mPos - mPosLast) / DeltaTime;
	vec3 mForce = vec3(0.0f, 0.0f, 0.0f);

	// calculate elasticity force for each neighbouring vertices
	for(int i = 0; i < 4; ++i)
	{
		int nID = int(roundEven(Neighbours[i]));
		vec3 nPos = vec3(InPosBuffer[nID]);
		vec3 nPosLast = vec3(InPosLastBuffer[nID]);
		vec3 nVel = (nPos - nPosLast) / DeltaTime;

		vec3 f = mPos - nPos;
		vec3 n = normalize(f);
		float fLength = length(f);
		float spring = fLength - SpringLengths[i];
		vec3 springiness = - Elasticity * spring * n;

		vec3 dV = mVel - nVel;
		float damp = ElDampCoeff * (dot(dV, f) / fLength);
		vec3 damping = damp * n;

		float sL = length(springiness);
		float dL = length(damping);
		damping = (damping / max(dL, 0.0000000001f)) * min(sL, dL);

		mForce = mForce + (springiness + damping) * NeighbourMultipliers[i];
	}

	// calculate gravity force
	float grav = Gravity * 0.1f;
	mForce = mForce + (Mass * vec3(0.0f, -grav, 0.0f));

	// calculate air damp force
	mForce = mForce + (-AirDampCoeff * mVel);

	// check hooks
	mForce = mForce * LockMultiplier;
	
	// calculate acceleration and use Verelet integration to calculate position
	vec3 newPos;
	vec3 acc = mForce / Mass;

	newPos = 2.0f * vec3(Pos) - vec3(PosLast) + acc * DeltaTime * DeltaTime;

	// update positions
	OutPos = vec4(newPos, 1.0f);
	OutPosLast = Pos;
	gl_Position = Pos;
}