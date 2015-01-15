#include "OrthoWindow.h"


OrthoWindow::OrthoWindow()
{
}

OrthoWindow::OrthoWindow(const OrthoWindow &other)
{
}


OrthoWindow::~OrthoWindow()
{
}

bool OrthoWindow::Initialize(ID3D11Device* device, unsigned int windowWidth, unsigned int windowHeight)
{
	bool result;

	mWindowWidth = windowWidth;
	mWindowHeight = windowHeight;

	// Initialize the vertex and index buffer that hold the geometry for the ortho window model.
	result = InitializeBuffers(device);
	if (!result)
	{
		return false;
	}

	return true;
}

OrthoWindow::VertexIndex* OrthoWindow::LoadGeometry(bool ind)
{
	m_vertexCount = 6;
	m_indexCount = 6;
	Vertex* vertices;
	unsigned long* indices;
	vertices = new Vertex[m_vertexCount];

	float left = (float)((mWindowWidth / 2)*-1);
	float right = left + (float)mWindowWidth;
	float top = (float)(mWindowHeight / 2);
	float bottom = top - (float)mWindowHeight;

	if (ind)
	{
		indices = new unsigned long[m_indexCount];
	}
	else indices = nullptr;

	// Load the vertex array with data.
	// First triangle.
	vertices[0].position = D3DXVECTOR3(left, top, 0.0f);  // Top left.
	vertices[0].texture = D3DXVECTOR2(0.0f, 0.0f);

	vertices[1].position = D3DXVECTOR3(right, bottom, 0.0f);  // Bottom right.
	vertices[1].texture = D3DXVECTOR2(1.0f, 1.0f);

	vertices[2].position = D3DXVECTOR3(left, bottom, 0.0f);  // Bottom left.
	vertices[2].texture = D3DXVECTOR2(0.0f, 1.0f);

	// Second triangle.
	vertices[3].position = D3DXVECTOR3(left, top, 0.0f);  // Top left.
	vertices[3].texture = D3DXVECTOR2(0.0f, 0.0f);

	vertices[4].position = D3DXVECTOR3(right, top, 0.0f);  // Top right.
	vertices[4].texture = D3DXVECTOR2(1.0f, 0.0f);

	vertices[5].position = D3DXVECTOR3(right, bottom, 0.0f);  // Bottom right.
	vertices[5].texture = D3DXVECTOR2(1.0f, 1.0f);

	if (ind)
	{
		// Load the index array with data.
		for (int i = 0; i<m_indexCount; i++)
		{
			indices[i] = i;
		}
	}

	VertexIndex* toReturn = new VertexIndex();
	toReturn->vertexArrayPtr = vertices;
	toReturn->indexArrayPtr = indices;
	return toReturn;
}