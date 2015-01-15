#pragma once

#define BUFFER_COUNT 2

#include <D3D11.h>
#include <D3DX10math.h>

class DeferredBuffer
{
private:
	int mTextureWidth, mTextureHeight;

	ID3D11Texture2D* mRenderTargetTextureArray[BUFFER_COUNT];
	ID3D11RenderTargetView* mRenderTargetViewArray[BUFFER_COUNT];
	ID3D11ShaderResourceView* mShaderResourceViewArray[BUFFER_COUNT];
	ID3D11Texture2D* mDepthStencilBuffer;
	ID3D11DepthStencilView* mDepthStencilView;
	D3D11_VIEWPORT mViewport;

public:
	DeferredBuffer();
	DeferredBuffer(const DeferredBuffer&);
	~DeferredBuffer();

	bool Initialize(ID3D11Device* device, int textureWidth, int textureHeight, float screenDepth, float screenNear);
	void Shutdown();

	void SetRenderTargets(ID3D11DeviceContext* deviceContext);
	void ClearRenderTargets(ID3D11DeviceContext* deviceContext, float r, float g, float b, float a);

	ID3D11ShaderResourceView* GetShaderResourceView(int number);
};

