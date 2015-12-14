#version 300 es

layout(location = 0) in vec4 Pos;					//current position
layout(location = 3) in vec4 Neighbours;			//neighbours id
layout(location = 6) in vec4 NeighbourMultipliers;	// neighbour multipliers (i.e. do I have to take it into consideration)
layout(location = 10) in vec4 Multipliers;			// x - lock muliplier, y - collision multiplier

uniform InPos
{
	vec4[16384] InPosBuffer;
};
uniform InBaaCols
{
	mat2x4[8192] baaBuffer;
};
uniform InSCols
{
	vec4[16384] sBuffer;
};

uniform mat4 WorldMatrix;
uniform float GroundLevel;
uniform int BoxAAColliderCount;
uniform int SphereColliderCount;
uniform int EdgesWidthAll;
uniform int EdgesLengthAll;
uniform int VertexCount;
uniform int InternalCollisionCheckWindowSize;

out vec4 OutPos;

void Vec3LengthSquared(in vec3 vec, out float ret)
{
	ret = vec.x * vec.x + vec.y * vec.y + vec.z * vec.z;
}

void CalculateCollisionSphere(vec3 mCenter, float mRadius, vec3 sphereCenter, float sphereRadius, float multiplier, inout vec3 ret)
{
	vec3 diff = mCenter - sphereCenter;
	float diffLength;
	Vec3LengthSquared(diff, diffLength);

	if
	(
		diffLength < (mRadius + sphereRadius) * (mRadius + sphereRadius) &&
		diffLength != 0.0f
	)
	{
		diff = normalize(diff);
		diff = diff * ((mRadius + sphereRadius) - sqrt(diffLength)) * multiplier;

		ret += diff;
	}
}

void CalculateCollisionBoxAA(vec3 mCenter, float mRadius, vec3 bMin, vec3 bMax, float multiplier, inout vec3 ret)
{
	vec3 closest = min(max(mCenter, bMin), bMax);
	float dist;
	Vec3LengthSquared(closest - mCenter, dist);

	if(dist < (mRadius * mRadius))
	{
		closest = mCenter - closest;
		ret += normalize(closest) * (mRadius - sqrt(dist)) * multiplier;
	}
}

void main()
{
	vec3 colOffset = vec3(0.0f, 0.0f, 0.0f);
	vec3 mPos = vec3(WorldMatrix * Pos);
	float mR = Multipliers.y;

	// solve external collisions
	for(int i = 0; i < BoxAAColliderCount; ++i)
	{
		mat2x4 box = baaBuffer[i];
		vec3 bMin = vec3(box[0][0], box[0][1], box[0][2]);
		vec3 bMax = vec3(box[1][0], box[1][1], box[1][2]);

		CalculateCollisionBoxAA(mPos, mR, bMin, bMax, 1.0f, colOffset);
	}

	for(int i = 0; i < SphereColliderCount; ++i)
	{
		vec4 sphere = sBuffer[i];
		vec3 sPos = vec3(sphere);
		float sR = sphere.w;

		CalculateCollisionSphere(mPos, mR, sPos, sR, 1.0f, colOffset);
	}

	// solve internal collisions
	//const int cc = 4;
	//const int nCount = (2 * cc + 1)*(2 * cc + 1) - 1;
	//int nIds[nCount];
	//int w = 0;
	//int v_cur = gl_VertexID;
	//for (int i = -cc; i <= cc; ++i)
	//{
	//	for (int j = -cc; j <= cc; ++j)
	//	{
	//		if (i == 0 && j == 0)
	//			continue;

	//		nIds[w] = abs((i * EdgesLengthAll + j + v_cur) % VertexCount);
	//		++w;
	//	}
	//}

	for (int i = 0; i < 4; ++i)
	{
		vec3 wnPos = vec3(WorldMatrix * InPosBuffer[int(Neighbours[i])]);
		CalculateCollisionSphere(mPos, mR, wnPos, mR, 0.5f, colOffset);
	}
	
	if(mPos.y < GroundLevel)
	{
		colOffset.y += (-mPos.y + GroundLevel);
	}

	//vec4 dupa = InPosBuffer[0];
	// update positions
	OutPos *= Multipliers.x;
	OutPos = vec4(Pos.x + colOffset.x, Pos.y + colOffset.y, Pos.z + colOffset.z, Pos.w);
	//OutPos = vec4(sBuffer[0][0], sBuffer[0][1], sBuffer[0][2], sBuffer[0][3]);
	gl_Position = Pos;
}