#version 300 es

layout(location = 0) in vec4 Pos;					//current position
layout(location = 1) in vec4 PosLast;				//previous position
layout(location = 2) in vec4 Normal;	
layout(location = 3) in vec4 Neighbours;			//neighbours id
layout(location = 4) in vec4 NeighboursDiag;
layout(location = 5) in vec4 Neighbours2;
layout(location = 6) in vec4 NeighbourMultipliers;	// neighbour multipliers (i.e. do I have to take it into consideration)
layout(location = 7) in vec4 NeighbourDiagMultipliers;
layout(location = 8) in vec4 Neighbour2Multipliers;
layout(location = 9) in vec4 ElMassCoeffs;			// x - elasticity, y - mass, z - el damp coeff, w - air damp coeff
layout(location = 10) in vec4 Multipliers;			// x - lock muliplier, y - collision multiplier

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
uniform vec4 SpringLengths;

out vec4 OutPos;
out vec4 OutPosLast;

void CalcDistConstraint(vec3 mPos, vec3 nPos, float mass, float sLength, float elCoeff, float dampCoeff, out vec4 constraint)
{
	elCoeff = clamp(elCoeff, 0.0f, 1.0f);
	vec3 diff = mPos - nPos;
	float cLength = length(diff);
	vec3 dP = (2.0f * mass) * (cLength - sLength) * (diff / cLength) * elCoeff;
	constraint.xyz = dP;
	constraint.w = 1.0f / mass;
}

void CalcBendConstraint(vec3 mPos, vec3 nPos1, vec3 nPos2, vec3 nPos3, float mass, float sLength, float elCoeff, float dampCoeff, out vec3 constraint)
{

}

void main()
{
	int mID = gl_VertexID;
	vec3 mPos = vec3(Pos);
	vec3 mPosLast = vec3(PosLast);
	vec3 mVel = (mPos - mPosLast) / DeltaTime;
	
	float sls1[4] = float[4](
		SpringLengths.y, SpringLengths.x, SpringLengths.y, SpringLengths.x
		);
	float sls2[4] = float[4](
		SpringLengths.z, SpringLengths.z, SpringLengths.z, SpringLengths.z
		);
	float sls3[4] = float[4](
		SpringLengths.y * SpringLengths.w, SpringLengths.x * SpringLengths.w, SpringLengths.y * SpringLengths.w, SpringLengths.x * SpringLengths.w
		);

	//////////////////////////////////////////////////////
	// forces calculation
	vec3 mForce = vec3(0.0f);
	vec3 posPredicted = vec3(0.0f);
		// calculate gravity force
	float grav = Gravity;
	mForce = mForce + (ElMassCoeffs.y * vec3(0.0f, -grav, 0.0f));

		// calculate air damp force
	mForce = mForce + (-ElMassCoeffs.w * mVel);

		// check hooks
	mForce = mForce * Multipliers.x;
	
	// calculate acceleration and use Verelet integration to calculate PREDICTED position
	vec3 acc = mForce / ElMassCoeffs.y;
	posPredicted = 2.0f * vec3(Pos) - vec3(PosLast) + acc * DeltaTime * DeltaTime;


	// compute constraints
	float elBias = 0.0005;
	vec3 cPos = vec3(0.0f);
	for(int i = 0; i < 4; ++i)
	{
		int nID = int(roundEven(Neighbours[i]));
		vec3 nPos = vec3(InPosBuffer[nID]);
		vec3 nPosLast = vec3(InPosLastBuffer[nID]);

		// distance constriant. XYZ is position, W is inverse of mass
		vec4 constraint;
		CalcDistConstraint(posPredicted, nPos, ElMassCoeffs.y, sls1[i], ElMassCoeffs.x * elBias, ElMassCoeffs.z * elBias, constraint);
		cPos -= constraint.xyz * constraint.w * NeighbourMultipliers[i];

		// bending constraint. Using the triangle bending constriant method. XYZ is position
		// currently impossible to implement due to per-vertex calculations method?
		//vec3 constraint;
	}

	for(int i = 0; i < 4; ++i)
	{
		int nID = int(roundEven(NeighboursDiag[i]));
		vec3 nPos = vec3(InPosBuffer[nID]);
		vec3 nPosLast = vec3(InPosLastBuffer[nID]);

		vec4 constraint;
		CalcDistConstraint(posPredicted, nPos, ElMassCoeffs.y, sls2[i], ElMassCoeffs.x * elBias, ElMassCoeffs.z * elBias, constraint);
		cPos -= constraint.xyz * constraint.w * NeighbourDiagMultipliers[i];
	}
	
	for(int i = 0; i < 4; ++i)
	{
		int nID = int(roundEven(Neighbours2[i]));
		vec3 nPos = vec3(InPosBuffer[nID]);
		vec3 nPosLast = vec3(InPosLastBuffer[nID]);

		vec4 constraint;
		CalcDistConstraint(posPredicted, nPos, ElMassCoeffs.y, sls3[i], ElMassCoeffs.x * elBias, ElMassCoeffs.z * elBias, constraint);
		cPos -= constraint.xyz * constraint.w * Neighbour2Multipliers[i];
	}

	// apply constraints
	vec3 finalPos = posPredicted + cPos * Multipliers.x;

	// update positions
	OutPos = vec4(finalPos, 1.0f);
	//OutPos = vec4(InPosBuffer[int(roundEven(NeighboursDiag[0]))][0], InPosBuffer[int(roundEven(NeighboursDiag[0]))][1], 
	//			InPosBuffer[int(roundEven(NeighboursDiag[0]))][2], InPosBuffer[int(roundEven(NeighboursDiag[0]))][3]);
	//OutPos = vec4(mForce, 1.0f);
	OutPosLast = Pos;
	gl_Position = Pos;
}