#pragma once

// includes
#include "Font.h"
#include "FontShader.h"

class Text
{
private:
	struct Sentence
	{
		ID3D11Buffer *vertexBuffer, *indexBuffer;
		int vertexCount, indexCount, maxLength;
		float r, g, b;
	};

	struct Vertex
	{
		D3DXVECTOR3 position;
		D3DXVECTOR2 texture;
	};

	Font* m_Font;
	FontShader* m_FontShader;

public:
	Text();
	Text(const Text&);
	~Text();
};

