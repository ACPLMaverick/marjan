#include "LightShader.h"


LightShader::LightShader()
{
	m_vertexShader = nullptr;
	m_pixelShader = nullptr;
	m_layout = nullptr;
	m_matrixBuffer = nullptr;
	m_sampleState = nullptr;
	m_lightBuffer = nullptr;
	m_ambientBuffer = nullptr;
}

LightShader::LightShader(const LightShader& other)
{
}

LightShader::~LightShader()
{
}

bool LightShader::Initialize(ID3D11Device* device, HWND hwnd, int id)
{
	bool result;
	this->myID = id;
	result = InitializeShader(device, hwnd, "DiffuseVertexShader.hlsl", "DiffusePixelShader.hlsl");
	if (!result) return false;
	return true;
}

bool LightShader::Render(ID3D11DeviceContext* deviceContext, int indexCount, D3DXMATRIX worldMatrix, D3DXMATRIX viewMatrix, 
	D3DXMATRIX projectionMatrix, ID3D11ShaderResourceView* texture, D3DXVECTOR4 diffuseColors[], D3DXVECTOR4 lightDirections[], unsigned int lightCount, D3DXVECTOR4 ambientColor)
{
	bool result;
	result = SetShaderParameters(deviceContext, worldMatrix, viewMatrix, projectionMatrix, texture, diffuseColors, lightDirections, lightCount, ambientColor);
	if (!result) return false;

	RenderShader(deviceContext, indexCount);

	return true;
}


bool LightShader::InitializeShader(ID3D11Device* device, HWND hwnd, LPCSTR vsFilename, LPCSTR psFilename)
{
	//////////////// loads the shader files and makes it usable to DirectX and the GPU

	HRESULT result;
	ID3D10Blob* errorMessage;
	ID3D10Blob* vertexShaderBuffer;
	ID3D10Blob* pixelShaderBuffer;
	D3D11_INPUT_ELEMENT_DESC polygonLayout[3];
	unsigned int numElements;
	D3D11_BUFFER_DESC matrixBufferDesc;
	D3D11_BUFFER_DESC lightBufferDesc;
	D3D11_BUFFER_DESC ambientBufferDesc;

	D3D11_SAMPLER_DESC samplerDesc;

	errorMessage = nullptr;
	vertexShaderBuffer = nullptr;
	pixelShaderBuffer = nullptr;

	// shader compilation into buffers

	result = D3DX11CompileFromFile(vsFilename, NULL, NULL, "DiffuseVertexShader", "vs_5_0", D3D10_SHADER_ENABLE_STRICTNESS, 0, NULL,
		&vertexShaderBuffer, &errorMessage, NULL);
	if (FAILED(result))
	{
		if (errorMessage)
		{
			OutputShaderErrorMessage(errorMessage, hwnd, vsFilename);
		}
		else
		{
			MessageBox(hwnd, vsFilename, "Missing Vertex Shader file!", MB_OK);
		}
		return false;
	}

	result = D3DX11CompileFromFile(psFilename, NULL, NULL, "DiffusePixelShader", "ps_5_0", D3D10_SHADER_ENABLE_STRICTNESS, 0, NULL,
		&pixelShaderBuffer, &errorMessage, NULL);
	if (FAILED(result))
	{
		if (errorMessage)
		{
			OutputShaderErrorMessage(errorMessage, hwnd, psFilename);
		}
		else
		{
			MessageBox(hwnd, psFilename, "Missing Pixel Shader file!", MB_OK);
		}
		return false;
	}

	// using buffers to create shader object themselves

	result = device->CreateVertexShader(vertexShaderBuffer->GetBufferPointer(), vertexShaderBuffer->GetBufferSize(), NULL, &m_vertexShader);
	if (FAILED(result)) return false;
	result = device->CreatePixelShader(pixelShaderBuffer->GetBufferPointer(), pixelShaderBuffer->GetBufferSize(), NULL, &m_pixelShader);
	if (FAILED(result)) return false;

	// setting up LAYOUT of data that goes into shader - matches Vertex type in Model class and Shader
	polygonLayout[0].SemanticName = "POSITION";
	polygonLayout[0].SemanticIndex = 0;
	polygonLayout[0].Format = DXGI_FORMAT_R32G32B32_FLOAT;
	polygonLayout[0].InputSlot = 0;
	polygonLayout[0].AlignedByteOffset = 0;			// important!
	polygonLayout[0].InputSlotClass = D3D11_INPUT_PER_VERTEX_DATA;
	polygonLayout[0].InstanceDataStepRate = 0;

	polygonLayout[1].SemanticName = "TEXCOORD";
	polygonLayout[1].SemanticIndex = 0;
	polygonLayout[1].Format = DXGI_FORMAT_R32G32_FLOAT;
	polygonLayout[1].InputSlot = 0;
	polygonLayout[1].AlignedByteOffset = D3D11_APPEND_ALIGNED_ELEMENT; // "border" between position and color - automatically solved by DX11
	polygonLayout[1].InputSlotClass = D3D11_INPUT_PER_VERTEX_DATA;
	polygonLayout[1].InstanceDataStepRate = 0;

	// normals layout
	polygonLayout[2].SemanticName = "NORMAL";
	polygonLayout[2].SemanticIndex = 0;
	polygonLayout[2].Format = DXGI_FORMAT_R32G32B32_FLOAT;
	polygonLayout[2].InputSlot = 0;
	polygonLayout[2].AlignedByteOffset = D3D11_APPEND_ALIGNED_ELEMENT;
	polygonLayout[2].InputSlotClass = D3D11_INPUT_PER_VERTEX_DATA;
	polygonLayout[2].InstanceDataStepRate = 0;

	// size of layout desc
	numElements = sizeof(polygonLayout) / sizeof(polygonLayout[0]);

	// creating input layout
	result = device->CreateInputLayout(polygonLayout, numElements, vertexShaderBuffer->GetBufferPointer(), vertexShaderBuffer->GetBufferSize(), &m_layout);
	if (FAILED(result)) return false;

	vertexShaderBuffer->Release();
	vertexShaderBuffer = nullptr;

	pixelShaderBuffer->Release();
	vertexShaderBuffer = nullptr;

	// setting vertex buffer as constant buffer
	matrixBufferDesc.Usage = D3D11_USAGE_DYNAMIC;	// because we want to refresh it every frame
	matrixBufferDesc.ByteWidth = sizeof(MatrixBuffer);
	matrixBufferDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;	// because I don't know :(
	matrixBufferDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
	matrixBufferDesc.MiscFlags = 0;
	matrixBufferDesc.StructureByteStride = 0;

	//creating pointer to constant buffer
	result = device->CreateBuffer(&matrixBufferDesc, NULL, &m_matrixBuffer);
	if (FAILED(result)) return false;

	// NEW - for texture sampler setup

	samplerDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
	samplerDesc.AddressU = D3D11_TEXTURE_ADDRESS_WRAP;
	samplerDesc.AddressV = D3D11_TEXTURE_ADDRESS_WRAP;
	samplerDesc.AddressW = D3D11_TEXTURE_ADDRESS_WRAP;
	samplerDesc.MipLODBias = 0.0f;
	samplerDesc.MaxAnisotropy = 1;
	samplerDesc.ComparisonFunc = D3D11_COMPARISON_ALWAYS;
	samplerDesc.BorderColor[0] = 0;
	samplerDesc.BorderColor[1] = 0;
	samplerDesc.BorderColor[2] = 0;
	samplerDesc.BorderColor[3] = 0;
	samplerDesc.MinLOD = 0;
	samplerDesc.MaxLOD = D3D11_FLOAT32_MAX;

	//create texture sampler state
	result = device->CreateSamplerState(&samplerDesc, &m_sampleState);
	if (FAILED(result)) return false;

	// Setup the description of the light dynamic constant buffer that is in the pixel shader.
	// Note that ByteWidth always needs to be a multiple of 16 if using D3D11_BIND_CONSTANT_BUFFER or CreateBuffer will fail.
	lightBufferDesc.Usage = D3D11_USAGE_DYNAMIC;
	lightBufferDesc.ByteWidth = sizeof(LightBuffer);
	lightBufferDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
	lightBufferDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
	lightBufferDesc.MiscFlags = 0;
	lightBufferDesc.StructureByteStride = 0;

	result = device->CreateBuffer(&lightBufferDesc, NULL, &m_lightBuffer);
	if (FAILED(result)) return false;

	ambientBufferDesc.Usage = D3D11_USAGE_DYNAMIC;
	ambientBufferDesc.ByteWidth = sizeof(AmbientBuffer);
	ambientBufferDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
	ambientBufferDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
	ambientBufferDesc.MiscFlags = 0;
	ambientBufferDesc.StructureByteStride = 0;

	result = device->CreateBuffer(&ambientBufferDesc, NULL, &m_ambientBuffer);
	if (FAILED(result)) return false;

	return true;
}


void LightShader::ShutdownShader()
{
	if (m_matrixBuffer)
	{
		m_matrixBuffer->Release();
		m_matrixBuffer = nullptr;
	}
	if (m_layout)
	{
		m_layout->Release();
		m_layout = nullptr;
	}
	if (m_pixelShader)
	{
		m_pixelShader->Release();
		m_pixelShader = nullptr;
	}
	if (m_vertexShader)
	{
		m_vertexShader->Release();
		m_vertexShader = nullptr;
	}
	if (m_sampleState)
	{
		m_sampleState->Release();
		m_sampleState = nullptr;
	}
	if (m_lightBuffer)
	{
		m_lightBuffer->Release();
		m_lightBuffer = nullptr;
	}
	if (m_ambientBuffer)
	{
		m_ambientBuffer->Release();
		m_ambientBuffer = nullptr;
	}
}

bool LightShader::SetShaderParameters(ID3D11DeviceContext* deviceContext, D3DXMATRIX worldMatrix, D3DXMATRIX viewMatrix, D3DXMATRIX projectionMatrix, 
	ID3D11ShaderResourceView* texture, D3DXVECTOR4 diffuseColors[], D3DXVECTOR4 lightDirections[], unsigned int lightCount, D3DXVECTOR4 ambientColor)
{
	HRESULT result;
	D3D11_MAPPED_SUBRESOURCE mappedResource;
	MatrixBuffer* dataPtr;
	unsigned int bufferNumber;
	LightBuffer* dataPtr02;
	AmbientBuffer* dataPtr03;

	// TRANSPOSING MATRICES!!!!!! REQUIREMENT IN DX11
	D3DXMatrixTranspose(&worldMatrix, &worldMatrix);
	D3DXMatrixTranspose(&viewMatrix, &viewMatrix);
	D3DXMatrixTranspose(&projectionMatrix, &projectionMatrix);

	// lock the constant buffer so it can be written to
	result = deviceContext->Map(m_matrixBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource);
	if (FAILED(result)) return false;

	// get the pointer to data in constant buffer
	dataPtr = (MatrixBuffer *)mappedResource.pData;

	// copy matrices
	dataPtr->world = worldMatrix;
	dataPtr->view = viewMatrix;
	dataPtr->projection = projectionMatrix;

	//unlock
	deviceContext->Unmap(m_matrixBuffer, 0);

	//set the matrices in vertex shader
	bufferNumber = 0;	// position of constant buffer inside the shader
	deviceContext->VSSetConstantBuffers(bufferNumber, 1, &m_matrixBuffer);

	// set shader resource in pixel shader
	deviceContext->PSSetShaderResources(0, 1, &texture);

	// setting the light constant buffer
	result = deviceContext->Map(m_lightBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource);
	if (FAILED(result)) return false;

	dataPtr02 = (LightBuffer*)mappedResource.pData;

	for (int i = 0; i < lightCount; i++)
	{
		dataPtr02->diffuseColor[i] = diffuseColors[i];
		dataPtr02->lightDirection[i] = lightDirections[i];
	}
	dataPtr02->lightDirection[0].w = lightCount;

	deviceContext->Unmap(m_lightBuffer, 0);

	result = deviceContext->Map(m_ambientBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource);
	if (FAILED(result)) return false;
	dataPtr03 = (AmbientBuffer*)mappedResource.pData;
	dataPtr03->ambientColor = ambientColor;
	deviceContext->Unmap(m_ambientBuffer, 0);

	bufferNumber = 0;
	ID3D11Buffer** buffers[2] = { &m_lightBuffer, &m_ambientBuffer };
	deviceContext->PSSetConstantBuffers(bufferNumber, 2, buffers[0]);

	return true;
}
