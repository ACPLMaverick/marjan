#pragma once
#ifndef _DIRECT3D_H_
#define _DIRECT3D_H_

// libs
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "d3dx11.lib")
#pragma comment(lib, "d3dx10.lib")

// includes
#include <DXGI.h>
#include <D3Dcommon.h>
#include <D3D11.h>
#include <D3DX10math.h>

class Direct3D
{
private:
	bool m_vsync_enabled;
	int m_videoCardMemory;
	char m_videoCardDescription[128];

	IDXGISwapChain* m_swapChain;
	ID3D11Device* m_device;
	ID3D11DeviceContext* m_deviceContext;
	ID3D11RenderTargetView* m_renderTargetView;
	ID3D11Texture2D* m_depthStencilBuffer;
	ID3D11DepthStencilState* m_depthStencilState;
	ID3D11DepthStencilView* m_depthStencilView;
	ID3D11RasterizerState* m_rasterState;
	ID3D11BlendState* m_blendState;

	D3DXMATRIX m_projectionMatrix;
	D3DXMATRIX m_worldMatrix;
	D3DXMATRIX m_orthoMatrix;
public:
	Direct3D();
	Direct3D(const Direct3D&);
	~Direct3D();

	bool Initialize(int, int, bool, HWND, bool, float, float);
	void Shutdown();

	void BeginScene(float, float, float, float);
	void EndScene();

	ID3D11Device* GetDevice();
	ID3D11DeviceContext* GetDeviceContext();

	void GetProjectionMatrix(D3DMATRIX&);
	void GetWorldnMatrix(D3DMATRIX&);
	void GetOrthoMatrix(D3DMATRIX&);

	void GetVideoCardInfo(char*, int&);
};

#endif

