#pragma once
#include <d3d11.h>
#include <d3dx10math.h>
#include <fstream>
using namespace std;

#include "Texture.h"

class Font
{
private:
	struct FontType
	{
		float left, right;
		int size;
	};

	struct Vertex
	{
		D3DXVECTOR3 position;
		D3DXVECTOR2 texture;
	};

	FontType* m_Font;
	Texture* m_Texture;

	const unsigned int charCount = 95;

	bool LoadFontData(char*);
	void ReleaseFontData();
	bool LoadTexture(ID3D11Device*, LPCSTR);
	void ReleaseTexture();
public:
	Font();
	Font(const Font&);
	~Font();

	virtual bool Initialize(ID3D11Device*, char*, LPCSTR);
	virtual void Shutdown();

	void BuildVertexArray(void*, char*, float, float);
	ID3D11ShaderResourceView* GetTexture();
};

