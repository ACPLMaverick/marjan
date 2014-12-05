#include "Sprite2D.h"


Sprite2D::Sprite2D() : Model()
{
}

Sprite2D::Sprite2D(D3DXVECTOR3 position, D3DXVECTOR3 rotation, D3DXVECTOR3 scale, D3D11_USAGE usage) : Model(position, rotation, scale, usage, "")
{
}

Sprite2D::~Sprite2D()
{
}

Sprite2D::VertexIndex* Sprite2D::LoadGeometry(bool ind)
{
	m_vertexCount = 4;
	m_indexCount = 6;
	Vertex* vertices;
	unsigned long* indices;
	vertices = new Vertex[m_vertexCount];
	
	if (ind)
	{
		indices = new unsigned long[m_indexCount];
	}
	else indices = nullptr;

	// rotation
	D3DXMATRIX rotateX;
	D3DXMATRIX rotateY;
	D3DXMATRIX rotateZ;
	D3DXMatrixRotationX(&rotateX, rotation.x);
	D3DXMatrixRotationY(&rotateY, rotation.y);
	D3DXMatrixRotationZ(&rotateZ, rotation.z);
	D3DXMATRIX rotationMatrix = rotateX*rotateY*rotateZ;
	D3DXVECTOR4 outputVec[4];



	// load vertex array with data
	vertices[0].position = D3DXVECTOR3(-1.0f, -1.0f, 0.0f); // BL
	vertices[0].texture = D3DXVECTOR2(0.0f, 1.0f);
	vertices[1].position = D3DXVECTOR3(-1.0f, 1.0f, 0.0f); // TL
	vertices[1].texture = D3DXVECTOR2(0.0f, 0.0f);
	vertices[2].position = D3DXVECTOR3(1.0f, -1.0f, 0.0f); // BR
	vertices[2].texture = D3DXVECTOR2(1.0f, 1.0f);
	vertices[3].position = D3DXVECTOR3(1.0f, 1.0f, 0.0f); // TR
	vertices[3].texture = D3DXVECTOR2(1.0f, 0.0f);

	for (int i = 0; i < 4; i++)
	{
		D3DXVec3Transform(&outputVec[i], &vertices[i].position, &rotationMatrix);
		vertices[i].position.x = (outputVec[i].x*scale.x + position.x);
		vertices[i].position.y = (outputVec[i].y*scale.y + position.y);
		vertices[i].position.z = (outputVec[i].z*scale.z + position.z);
	}

	if (ind)
	{
		// load index array with data
		indices[0] = 0; // BL
		indices[1] = 1; // TL
		indices[2] = 2; // BR
		indices[3] = 2;
		indices[4] = 1;
		indices[5] = 3;
	}

	VertexIndex* toReturn = new VertexIndex();
	toReturn->vertexArrayPtr = vertices;
	toReturn->indexArrayPtr = indices;
	return toReturn;
}
