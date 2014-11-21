#include "Model3D.h"


Model3D::Model3D() : Model()
{
}

Model3D::Model3D(const Model3D& other) : Model(other)
{
}

Model3D::Model3D(D3DXVECTOR3 position, D3DXVECTOR3 rotation, D3DXVECTOR3 scale, D3D11_USAGE usage, string filePath)
	: Model(position, rotation, scale, usage)
{

}

Model3D::~Model3D()
{
}

Model3D::VertexIndex* Model3D::LoadGeometry(bool ind)
{
	Vertex* vertices;
	unsigned long* indices;

	m_vertexCount = 8;
	m_indexCount = 36;

	vertices = new Vertex[m_vertexCount];
	if (!vertices)
	{
		return false;
	}

	indices = new unsigned long[m_indexCount];
	if (!indices)
	{
		return false;
	}


	D3DXMATRIX rotationMatrix;

	float pitch = rotation.x * 0.0174532925f;
	float yaw = rotation.y * 0.0174532925f;
	float roll = rotation.z * 0.0174532925f;

	D3DXMatrixRotationYawPitchRoll(&rotationMatrix, yaw, pitch, roll);

	D3DXVECTOR3 pos;

	vertices[0].position = D3DXVECTOR3(-scale.x, scale.y, -scale.z);// +m_position;
	vertices[0].texture = D3DXVECTOR2(0.0f, 0.5f);
	/*vertices[0].normal = D3DXVECTOR3(-1.0f, 1.0f, 1.0f);*/

	vertices[1].position = D3DXVECTOR3(scale.x, scale.y, -scale.z);// +m_position;
	vertices[1].texture = D3DXVECTOR2(0.0f, 0.0f);
	//vertices[1].normal = D3DXVECTOR3(1.0f, 1.0f, 1.0f);

	vertices[2].position = D3DXVECTOR3(scale.x, scale.y, scale.z);// +m_position;
	vertices[2].texture = D3DXVECTOR2(0.5f, 0.0f);
	//vertices[2].normal = D3DXVECTOR3(1.0f, 1.0f, -1.0f);

	vertices[3].position = D3DXVECTOR3(-scale.x, scale.y, scale.z);// +m_position;
	vertices[3].texture = D3DXVECTOR2(0.5f, 0.5f);
	//vertices[3].normal = D3DXVECTOR3(-1.0f, 1.0f, -1.0f);

	vertices[4].position = D3DXVECTOR3(-scale.x, -scale.y, -scale.z);// +m_position;
	vertices[4].texture = D3DXVECTOR2(0.5f, 1.0f);
	//vertices[4].normal = D3DXVECTOR3(-1.0f, -1.0f, 1.0f);

	vertices[5].position = D3DXVECTOR3(scale.x, -scale.y, -scale.z);// +m_position;
	vertices[5].texture = D3DXVECTOR2(0.5f, 0.5f);
	//vertices[5].normal = D3DXVECTOR3(1.0f, -1.0f, 1.0f);

	vertices[6].position = D3DXVECTOR3(scale.x, -scale.y, scale.z);// +m_position;
	vertices[6].texture = D3DXVECTOR2(1.0f, 0.5f);
	//vertices[6].normal = D3DXVECTOR3(1.0f, -1.0f, -1.0f);

	vertices[7].position = D3DXVECTOR3(-scale.x, -scale.y, scale.z);// +m_position;
	vertices[7].texture = D3DXVECTOR2(1.0f, 1.0f);
	//vertices[7].normal = D3DXVECTOR3(-1.0f, -1.0f, -1.0f);

	for (int i = 0; i < m_vertexCount; i++)
	{
		D3DXVec3TransformCoord(&pos, &vertices[i].position, &rotationMatrix);
		vertices[i].position = pos + position;
	}

	indices[0] = 3;
	indices[1] = 1;
	indices[2] = 0;
	indices[3] = 2;
	indices[4] = 1;
	indices[5] = 3;
	indices[6] = 0;
	indices[7] = 5;
	indices[8] = 4;
	indices[9] = 1;
	indices[10] = 5;
	indices[11] = 0;
	indices[12] = 3;
	indices[13] = 4;
	indices[14] = 7;
	indices[15] = 0;
	indices[16] = 4;
	indices[17] = 3;
	indices[18] = 1;
	indices[19] = 6;
	indices[20] = 5;
	indices[21] = 2;
	indices[22] = 6;
	indices[23] = 1;
	indices[24] = 2;
	indices[25] = 7;
	indices[26] = 6;
	indices[27] = 3;
	indices[28] = 7;
	indices[29] = 2;
	indices[30] = 6;
	indices[31] = 4;
	indices[32] = 5;
	indices[33] = 7;
	indices[34] = 4;
	indices[35] = 6;

	VertexIndex* toRet = new VertexIndex;
	toRet->vertexArrayPtr = vertices;
	toRet->indexArrayPtr = indices;

	return toRet;
}
