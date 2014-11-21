#include "Text.h"


Text::Text()
{
	m_Font = nullptr;
	m_FontShader = nullptr;
	sen01 = nullptr;
	sen02 = nullptr;
}

Text::Text(const Text &other)
{

}

Text::~Text()
{
}

bool Text::Initialize(ID3D11Device* device, ID3D11DeviceContext* deviceContext,
	HWND hwnd, int screenWidth, int screenHeight, D3DXMATRIX baseViewMatrix)
{
	bool result;
	m_screenWidth = screenWidth;
	m_screenHeight = screenHeight;
	m_baseViewMatrix = baseViewMatrix;


	m_Font = new Font();
	result = m_Font->Initialize(device, fontCfgPath, fontImgPath.c_str());
	if (!result)
	{
		MessageBox(hwnd, "Could not initialize the font object", "Error", MB_OK);
		return false;
	}

	m_FontShader = new FontShader();
	result = m_FontShader->Initialize(device, hwnd);
	if (!result)
	{
		MessageBox(hwnd, "Could not initialize the font shader object", "Error", MB_OK);
		return false;
	}


	result = InitializeSentence(&sen01, 16, device);
	if (!result) return false;

	result = InitializeSentence(&sen02, 16, device);
	if (!result) return false;

	result = UpdateSentence(sen01, "chuj, dupa", 0, 0, 1.0f, 1.0f, 1.0f, deviceContext);
	if (!result) return false;

	result = UpdateSentence(sen02, "i kamieni kupa", 0, 0, 1.0f, 1.0f, 1.0f, deviceContext);
	if (!result) return false;

	return true;
}

void Text::Shutdown()
{
	ReleaseSentence(&sen01);
	ReleaseSentence(&sen02);

	if (m_FontShader)
	{
		m_FontShader->Shutdown();
		delete m_FontShader;
		m_FontShader = nullptr;
	}

	if (m_Font)
	{
		m_Font->Shutdown();
		delete m_Font;
		m_Font = nullptr;
	}
}

bool Text::Render(ID3D11DeviceContext* deviceContext, D3DXMATRIX worldMatrix, D3DXMATRIX orthoMatrix)
{
	bool result;

	result = RenderSentence(deviceContext, sen01, worldMatrix, orthoMatrix);
	if (!result) return false;
	result = RenderSentence(deviceContext, sen02, worldMatrix, orthoMatrix);
	if (!result) return true;
}

bool Text::InitializeSentence(Sentence** sentence, int maxLength, ID3D11Device* device)
{
	Vertex* vertices;
	unsigned long* indices;
	D3D11_BUFFER_DESC vertexBufferDesc, indexBufferDesc;
	D3D11_SUBRESOURCE_DATA vertexData, indexData;
	HRESULT result;
	
	*sentence = new Sentence();

	(*sentence)->vertexBuffer = 0;
	(*sentence)->indexBuffer = 0;
	(*sentence)->maxLength = maxLength;
	(*sentence)->vertexCount = 6 * maxLength;
	(*sentence)->indexCount = (*sentence)->vertexCount;

	vertices = new Vertex[(*sentence)->vertexCount];
	indices = new unsigned long[(*sentence)->indexCount];
	for (int i = 0; i < (*sentence)->indexCount; i++)
	{
		indices[i] = i;
	}

	// Set up the description of the dynamic vertex buffer.
	vertexBufferDesc.Usage = D3D11_USAGE_DYNAMIC;
	vertexBufferDesc.ByteWidth = sizeof(Vertex) * (*sentence)->vertexCount;
	vertexBufferDesc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
	vertexBufferDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
	vertexBufferDesc.MiscFlags = 0;
	vertexBufferDesc.StructureByteStride = 0;

	// Give the subresource structure a pointer to the vertex data.
	vertexData.pSysMem = vertices;
	vertexData.SysMemPitch = 0;
	vertexData.SysMemSlicePitch = 0;

	// Create the vertex buffer.
	result = device->CreateBuffer(&vertexBufferDesc, &vertexData, &(*sentence)->vertexBuffer);
	if (FAILED(result))
	{
		return false;
	}

	// Set up the description of the static index buffer.
	indexBufferDesc.Usage = D3D11_USAGE_DEFAULT;
	indexBufferDesc.ByteWidth = sizeof(unsigned long)* (*sentence)->indexCount;
	indexBufferDesc.BindFlags = D3D11_BIND_INDEX_BUFFER;
	indexBufferDesc.CPUAccessFlags = 0;
	indexBufferDesc.MiscFlags = 0;
	indexBufferDesc.StructureByteStride = 0;

	// Give the subresource structure a pointer to the index data.
	indexData.pSysMem = indices;
	indexData.SysMemPitch = 0;
	indexData.SysMemSlicePitch = 0;

	// Create the index buffer.
	result = device->CreateBuffer(&indexBufferDesc, &indexData, &(*sentence)->indexBuffer);
	if (FAILED(result))
	{
		return false;
	}

	// Release the vertex array as it is no longer needed.
	delete[] vertices;
	vertices = 0;

	// Release the index array as it is no longer needed.
	delete[] indices;
	indices = 0;

	return true;
}

bool Text::UpdateSentence(Sentence* sentence, char* text, int posX, int posY, 
	float r, float g, float b, ID3D11DeviceContext* deviceContext)
{
	int numLetters;
	Vertex* vertices;
	float drawX, drawY;
	HRESULT result;
	D3D11_MAPPED_SUBRESOURCE mappedResource;
	Vertex* verticesPtr;

	// set the color and size of the sentence
	sentence->r = r;
	sentence->g = g;
	sentence->b = b;

	numLetters = (int)strlen(text);
	if (numLetters > sentence->maxLength) return false;

	vertices = new Vertex[sentence->vertexCount];
	if (!vertices) return false;

	// initialize vertex array to zeros at first
	memset(vertices, 0, (sizeof(Vertex)*sentence->vertexCount));

	// calculate X and Y pixel position on the screen to start drawing to
	drawX = (float)(((m_screenWidth/2)*-1) + posX);
	drawY = (float)((m_screenWidth / 2) + posY);

	// build the vertex array from the sentence text and draw loc
	m_Font->BuildVertexArray((void*)vertices, text, drawX, drawY);

	result = deviceContext->Map(sentence->vertexBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource);
	if (FAILED(result)) return false;

	verticesPtr = (Vertex*)mappedResource.pData;
	memcpy(verticesPtr, (void*)vertices, (sizeof(Vertex)*sentence->vertexCount));

	deviceContext->Unmap(sentence->vertexBuffer, 0);

	delete[] vertices;
	vertices = 0;

	return true;
}

void Text::ReleaseSentence(Sentence** sentence)
{
	if (*sentence)
	{
		if ((*sentence)->vertexBuffer)
		{
			(*sentence)->vertexBuffer->Release();
			(*sentence)->vertexBuffer = 0;
		}

		if ((*sentence)->indexBuffer)
		{
			(*sentence)->indexBuffer->Release();
			(*sentence)->indexBuffer = 0;
		}

		delete *sentence;
		*sentence = 0;
	}
}

bool Text::RenderSentence(ID3D11DeviceContext* deviceContext, Sentence* sentence,
	D3DXMATRIX worldMatrix, D3DXMATRIX orthoMatrix)
{
	unsigned int stride, offset;
	D3DXVECTOR4 pixelColor;
	bool result;

	stride = sizeof(Vertex);
	offset = 0;

	deviceContext->IASetVertexBuffers(0, 1, &sentence->vertexBuffer, &stride, &offset);
	deviceContext->IASetIndexBuffer(sentence->indexBuffer, DXGI_FORMAT_R32_UINT, 0);
	deviceContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

	pixelColor = D3DXVECTOR4(sentence->r, sentence->g, sentence->b, 1.0f);
	result = m_FontShader->Render(deviceContext, sentence->indexCount, worldMatrix,
		m_baseViewMatrix, orthoMatrix, m_Font->GetTexture(), pixelColor);
	if (!result) false;
	return true;
}