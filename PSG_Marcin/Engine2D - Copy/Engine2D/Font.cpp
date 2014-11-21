#include "Font.h"


Font::Font()
{
	m_Font = nullptr;
	m_Texture = nullptr;
}

Font::Font(const Font& other)
{

}

Font::~Font()
{
}

bool Font::Initialize(ID3D11Device* device, char* configPath, LPCSTR texturePath)
{
	bool result;
	result = LoadFontData(configPath);
	if (!result) return false;

	result = LoadTexture(device, texturePath);
	if (!result) return false;
	return true;
}

void Font::Shutdown()
{
	ReleaseTexture();
	ReleaseFontData();
}

void Font::BuildVertexArray(void* vertices, char* sentence, float drawX, float drawY)
{
	Vertex* vertexPtr = (Vertex*)vertices;
	int numLetters = (int)strlen(sentence);
	int index = 0;
	int letter;

	for (int i = 0; i < numLetters; i++)
	{
		letter = (int)sentence[i] - 32;

		// check if space
		if (letter == 0) drawX += 3.0f;
		else
		{
			vertexPtr[index].position = D3DXVECTOR3(drawX, drawY, 0.0f);
			vertexPtr[index].texture = D3DXVECTOR2(m_Font[letter].left, 1.0f);
			index++;

			vertexPtr[index].position = D3DXVECTOR3((drawX + m_Font[letter].size), (drawY - 16.0f), 0.0f);
			vertexPtr[index].texture = D3DXVECTOR2(m_Font[letter].right, 1.0f);
			index++;

			vertexPtr[index].position = D3DXVECTOR3(drawX, (drawY - 16.0f), 0.0f);
			vertexPtr[index].texture = D3DXVECTOR2(m_Font[letter].left, 1.0f);
			index++;

			vertexPtr[index].position = D3DXVECTOR3(drawX, drawY, 0.0f);
			vertexPtr[index].texture = D3DXVECTOR2(m_Font[letter].left, 0.0f);
			index++;

			vertexPtr[index].position = D3DXVECTOR3((drawX + m_Font[letter].size), drawY, 0.0f);
			vertexPtr[index].texture = D3DXVECTOR2(m_Font[letter].right, 0.0f);
			index++;

			vertexPtr[index].position = D3DXVECTOR3((drawX + m_Font[letter].size), (drawY - 16.0f), 0.0f);
			vertexPtr[index].texture = D3DXVECTOR2(m_Font[letter].right, 0.0f);
			index++;

			drawX = drawX + m_Font[letter].size + 1.0f;
		}
	}
}

ID3D11ShaderResourceView* Font::GetTexture()
{
	return m_Texture->GetTexture();
}

bool Font::LoadFontData(char* path)
{
	ifstream input;
	char temp;

	m_Font = new FontType[charCount];
	input.open(path);
	if (input.fail()) return false;

	for (int i = 0; i < charCount; i++)
	{
		input.get(temp);
		while (temp != ' ') input.get(temp);
		input.get(temp);
		while (temp != ' ') input.get(temp);
		input >> m_Font[i].left;
		input >> m_Font[i].right;
		input >> m_Font[i].size;
	}
	input.close();
	return true;
}

void Font::ReleaseFontData()
{
	if (m_Font)
	{
		delete[] m_Font;
		m_Font = 0;
	}
}

bool Font::LoadTexture(ID3D11Device* device, LPCSTR path)
{
	bool result;
	m_Texture = new Texture();

	result = m_Texture->Initialize(device, path);
	return result;
}

void Font::ReleaseTexture()
{
	if (m_Texture)
	{
		m_Texture->Shutdown();
		delete m_Texture;
	}
}
