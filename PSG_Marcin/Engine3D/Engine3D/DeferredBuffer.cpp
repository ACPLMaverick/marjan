#include "DeferredBuffer.h"


DeferredBuffer::DeferredBuffer()
{
	for (int i = 0; i < BUFFER_COUNT; i++)
	{
		mRenderTargetTextureArray[i] = nullptr;
		mRenderTargetViewArray[i] = nullptr;
		mShaderResourceViewArray[i] = nullptr;
	}
	mDepthStencilBuffer = nullptr;
	mDepthStencilView = nullptr;
}

DeferredBuffer::DeferredBuffer(const DeferredBuffer& other)
{
}

DeferredBuffer::~DeferredBuffer()
{
}

bool DeferredBuffer::Initialize(ID3D11Device* device, int textureWidth, int textureHeight, float screenDepth, float screenNear)
{
	D3D11_TEXTURE2D_DESC textureDesc;
	HRESULT result;
	D3D11_RENDER_TARGET_VIEW_DESC renderTargetViewDesc;
	D3D11_SHADER_RESOURCE_VIEW_DESC shaderResourceViewDesc;
	D3D11_TEXTURE2D_DESC depthBufferDesc;
	D3D11_DEPTH_STENCIL_VIEW_DESC depthStencilViewDesc;

	mTextureHeight = textureHeight;
	mTextureWidth = textureWidth;

	ZeroMemory(&textureDesc, sizeof(textureDesc));

	// setup the render target texture description
	textureDesc.Width = textureWidth;
	textureDesc.Height = textureHeight;
	textureDesc.MipLevels = 1;
	textureDesc.ArraySize = 1;
	textureDesc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
	textureDesc.SampleDesc.Count = 1;
	textureDesc.Usage = D3D11_USAGE_DEFAULT;
	textureDesc.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
	textureDesc.CPUAccessFlags = 0;
	textureDesc.MiscFlags = 0;

	// create array of render target textures for the deferred shader to write to
	for (int i = 0; i < BUFFER_COUNT; i++)
	{
		result = device->CreateTexture2D(&textureDesc, NULL, &mRenderTargetTextureArray[i]);
		if (FAILED(result)) return false;
	}

	// setup the description for render target view
	renderTargetViewDesc.Format = textureDesc.Format;
	renderTargetViewDesc.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2D;
	renderTargetViewDesc.Texture2D.MipSlice = 0;

	// create the render target views
	for (int i = 0; i < BUFFER_COUNT; i++)
	{
		result = device->CreateRenderTargetView(mRenderTargetTextureArray[i], &renderTargetViewDesc, &mRenderTargetViewArray[i]);
		if (FAILED(result)) return false;
	}

	// setup the description of shader resource view
	shaderResourceViewDesc.Format = textureDesc.Format;
	shaderResourceViewDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
	shaderResourceViewDesc.Texture2D.MostDetailedMip = 0;
	shaderResourceViewDesc.Texture2D.MipLevels = 1;

	// create the shader resource views
	for (int i = 0; i < BUFFER_COUNT; i++)
	{
		result = device->CreateShaderResourceView(mRenderTargetTextureArray[i], &shaderResourceViewDesc, &mShaderResourceViewArray[i]);
		if (FAILED(result)) return false;
	}

	ZeroMemory(&depthBufferDesc, sizeof(depthBufferDesc));

	// DEPTH BUFFER DESCRIPTION
	depthBufferDesc.Width = textureWidth;
	depthBufferDesc.Height = textureHeight;
	depthBufferDesc.MipLevels = 1;
	depthBufferDesc.ArraySize = 1;
	depthBufferDesc.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
	depthBufferDesc.SampleDesc.Count = 1;
	depthBufferDesc.SampleDesc.Quality = 0;
	depthBufferDesc.Usage = D3D11_USAGE_DEFAULT;
	depthBufferDesc.BindFlags = D3D11_BIND_DEPTH_STENCIL;
	depthBufferDesc.CPUAccessFlags = 0;
	depthBufferDesc.MiscFlags = 0;

	// create texture for depth buffer
	result = device->CreateTexture2D(&depthBufferDesc, NULL, &mDepthStencilBuffer);
	if (FAILED(result)) return false;

	// now the depth stencil and its description
	depthStencilViewDesc.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
	depthStencilViewDesc.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;
	depthStencilViewDesc.Texture2D.MipSlice = 0;

	result = device->CreateDepthStencilView(mDepthStencilBuffer, &depthStencilViewDesc, &mDepthStencilView);
	if (FAILED(result)) return false;

	// setup viewport for rendering
	mViewport.Width = (float)textureWidth;
	mViewport.Height = (float)textureHeight;
	mViewport.MinDepth = 0.0f;
	mViewport.MaxDepth = 1.0f;
	mViewport.TopLeftX = 0.0f;
	mViewport.TopLeftY = 0.0f;

	return true;
}

void DeferredBuffer::Shutdown()
{
	if (mDepthStencilView)
	{
		mDepthStencilView->Release();
		mDepthStencilView = nullptr;
	}
	if (mDepthStencilBuffer)
	{
		mDepthStencilBuffer->Release();
		mDepthStencilBuffer = nullptr;
	}

	for (int i = 0; i < BUFFER_COUNT; i++)
	{
		if (mShaderResourceViewArray[i])
		{
			mShaderResourceViewArray[i]->Release();
			mShaderResourceViewArray[i] = nullptr;
		}
		if (mRenderTargetViewArray[i])
		{
			mRenderTargetViewArray[i]->Release();
			mRenderTargetViewArray[i] = nullptr;
		}
		if (mRenderTargetTextureArray[i])
		{
			mRenderTargetTextureArray[i]->Release();
			mRenderTargetTextureArray[i] = nullptr;
		}
	}
}

void DeferredBuffer::SetRenderTargets(ID3D11DeviceContext* deviceContext)
{
	// Bind the render target view array and depth stencil buffer to the output render pipeline
	deviceContext->OMSetRenderTargets(BUFFER_COUNT, mRenderTargetViewArray, mDepthStencilView);

	// set teh viewport
	deviceContext->RSSetViewports(1, &mViewport);
}

void DeferredBuffer::ClearRenderTargets(ID3D11DeviceContext* deviceContext, float r, float g, float b, float a)
{
	// clears all of the render targer buffers in this object
	float color[4] = { r, g, b, a };

	for (int i = 0; i < BUFFER_COUNT; i++)
	{
		deviceContext->ClearRenderTargetView(mRenderTargetViewArray[i], color);
	}

	deviceContext->ClearDepthStencilView(mDepthStencilView, D3D11_CLEAR_DEPTH, 1.0f, 0);
}

ID3D11ShaderResourceView* DeferredBuffer::GetShaderResourceView(int number)
{
	return mShaderResourceViewArray[number];
}