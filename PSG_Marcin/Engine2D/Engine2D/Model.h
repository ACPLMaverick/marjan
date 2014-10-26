#pragma once

// includes
#include <d3d11.h>
#include <d3dx10math.h>
#include <string>
using namespace std;

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

	ID3D11Device* myDevice;
	ID3D11Buffer *m_vertexBuffer, *m_indexBuffer;
	int m_vertexCount, m_indexCount;
	D3D11_BUFFER_DESC vertexBufferDesc, indexBufferDesc;
	D3D11_SUBRESOURCE_DATA vertexData, indexData;

	Texture* m_texture;

	bool InitializeBuffers(ID3D11Device*);
	void UpdateBuffers(ID3D11Device*);
	void ShutdownBuffers();
	void RenderBuffers(ID3D11DeviceContext*);

	bool LoadTexture(ID3D11Device*, LPCSTR);
	void ReleaseTexture();

	virtual VertexIndex* LoadGeometry() = 0;
public:
	D3DXVECTOR3 position;
	D3DXVECTOR3 rotation;
	D3DXVECTOR3 scale;

	Model();
	Model(D3DXVECTOR3 position, D3DXVECTOR3 rotation, D3DXVECTOR3 scale);
	Model(const Model&);
	~Model();

	bool Initialize(ID3D11Device*, Texture* texture);
	void Shutdown();
	void Render(ID3D11DeviceContext*);

	int GetIndexCount();
	ID3D11ShaderResourceView* GetTexture();
};

