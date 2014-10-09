#pragma once

// includes
#include <d3d11.h>
#include <d3dx10math.h>

class Model
{
private:
	struct Vertex
	{
		D3DXVECTOR3 position;
		D3DXVECTOR4 color;
	};

	struct VertexIndex
	{
		Vertex* vertexArrayPtr;
		unsigned long* indexArrayPtr;
	};

	ID3D11Buffer *m_vertexBuffer, *m_indexBuffer;
	int m_vertexCount, m_indexCount;

	bool InitializeBuffers(ID3D11Device*);
	void ShutdownBuffers();
	void RenderBuffers(ID3D11DeviceContext*);

	virtual VertexIndex LoadGeometry();
public:
	Model();
	Model(const Model&);
	~Model();

	bool Initialize(ID3D11Device*);
	void Shutdown();
	void Render(ID3D11DeviceContext*);

	int GetIndexCount();
};

