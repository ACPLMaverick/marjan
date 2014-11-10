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
	int m_screenWidth, m_screenHeight;
	D3DXMATRIX m_baseViewMatrix;

	Sentence* sen01;
	Sentence* sen02;

	bool InitializeSentence(Sentence**, int, ID3D11Device*);
	bool UpdateSentence(Sentence*, char*, int, int, float, float, float, ID3D11DeviceContext*);
	void ReleaseSentence(Sentence**);
	bool RenderSentence(ID3D11DeviceContext*, Sentence*, D3DXMATRIX, D3DXMATRIX);

public:
	char* fontCfgPath = "./Configs/FontBasic.cfg";
	const string fontImgPath = "./Textures/FontBasic.cfg";

	Text();
	Text(const Text&);
	~Text();

	bool Initialize(ID3D11Device*, ID3D11DeviceContext*, HWND, int, int, D3DXMATRIX);
	void Shutdown();
	bool Render(ID3D11DeviceContext*, D3DXMATRIX, D3DXMATRIX);
};

