#ifndef _TEXTURECLASS_H_
#define _TEXTURECLASS_H_

#include <D3D11.h>
#include <D3DX11tex.h>

class TextureClass
{
public:
	TextureClass();

	bool Initialize(ID3D11Device*, WCHAR*);
	void Shutdown();
	ID3D11ShaderResourceView* GetTexture(); /*returns a pointer to the texture resource so that it can be used for rendering by shaders.*/

private:
	ID3D11ShaderResourceView* m_texture;
};

#endif _TEXTURECLASS_H_