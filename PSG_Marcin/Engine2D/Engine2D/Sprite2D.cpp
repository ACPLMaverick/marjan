#include "Sprite2D.h"


Sprite2D::Sprite2D() : Model()
{
}

Sprite2D::Sprite2D(D3DXVECTOR3 position, D3DXVECTOR3 rotation, D3DXVECTOR3 scale) : Model(position, rotation, scale)
{
}

Sprite2D::~Sprite2D()
{
}

Sprite2D::VertexIndex Sprite2D::LoadGeometry()
{
	Vertex* vertices;
	unsigned long* indices;
	
	m_vertexCount = 4;
	m_indexCount = 6;

	vertices = new Vertex[m_vertexCount];
	indices = new unsigned long[m_indexCount];

	// load vertex array with data
	vertices[0].position = D3DXVECTOR3(-1.0f, -1.0f, 0.0f) + position; // BL
	vertices[0].texture = D3DXVECTOR2(0.0f, 1.0f);
	vertices[1].position = D3DXVECTOR3(-1.0f, 1.0f, 0.0f) + position; // TL
	vertices[1].texture = D3DXVECTOR2(0.0f, 0.0f);
	vertices[2].position = D3DXVECTOR3(1.0f, -1.0f, 0.0f) + position; // BR
	vertices[2].texture = D3DXVECTOR2(1.0f, 1.0f);
	vertices[3].position = D3DXVECTOR3(1.0f, 1.0f, 0.0f) + position; // TR
	vertices[3].texture = D3DXVECTOR2(1.0f, 0.0f);

	// load index array with data
	indices[0] = 0; // BL
	indices[1] = 1; // TL
	indices[2] = 2; // BR
	indices[3] = 2;
	indices[4] = 1;
	indices[5] = 3;

	VertexIndex toReturn;
	toReturn.vertexArrayPtr = vertices;
	toReturn.indexArrayPtr = indices;
	return toReturn;
}
