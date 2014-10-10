#pragma once

// includes
#include <d3d11.h>
#include <d3dx10math.h>

// my classes
#include "Texture.h"

class Model
{
protected:
	struct Vertex
	{
		D3DXVECTOR3 position;
		D3DXVECTOR2 texture;
	};

	struct VertexIndex
	{
		Vertex* vertexArrayPtr;
		unsigned long* indexArrayPtr;
	};

	ID3D11Buffer *m_vertexBuffer, *m_indexBuffer;
	int m_vertexCount, m_indexCount;

	D3DXVECTOR3 position;

	Texture* m_texture;

	bool InitializeBuffers(ID3D11Device*);
	void UpdateBuffers(ID3D11Device*);
	void ShutdownBuffers();
	void RenderBuffers(ID3D11DeviceContext*);

	bool LoadTexture(ID3D11Device*, LPCSTR);
	void ReleaseTexture();

	virtual VertexIndex LoadGeometry();
public:
	Model();
	Model(D3DXVECTOR3 position);
	Model(const Model&);
	~Model();

	bool Initialize(ID3D11Device*, LPCSTR texFilename);
	void Shutdown();
	void Render(ID3D11DeviceContext*);

	int GetIndexCount();
	ID3D11ShaderResourceView* GetTexture();
};

