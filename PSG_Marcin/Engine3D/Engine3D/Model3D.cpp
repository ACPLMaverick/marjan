#include "Model3D.h"


Model3D::Model3D() : Model()
{
}

Model3D::Model3D(const Model3D& other) : Model(other)
{
}

Model3D::Model3D(D3DXVECTOR3 position, D3DXVECTOR3 rotation, D3DXVECTOR3 scale, D3D11_USAGE usage, string filePath)
	: Model(position, rotation, scale, usage, filePath)
{
	myGeometry = nullptr;
}

Model3D::~Model3D()
{
	if(myGeometry != nullptr) delete myGeometry;
}

Model3D::VertexIndex* Model3D::LoadGeometry(bool ind, string filePath)
{
	if (myGeometry != nullptr)
	{
		UpdateGeometry();
		return myGeometry;
	}

	Vertex* vertices;
	unsigned long* indices;
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> materials;

	string error = tinyobj::LoadObj(shapes, materials, filePath.c_str(), NULL);

	m_vertexCount = shapes.at(0).mesh.positions.size() / 3;
	m_indexCount = shapes.at(0).mesh.indices.size();

	vertices = new Vertex[m_vertexCount];
	if (!vertices)
	{
		return false;
	}
	for (int i = 0, j = 0, k = 0; i < m_vertexCount; i++, j+=3, k+=2)
	{
		vertices[i].position.x = shapes.at(0).mesh.positions.at(j);
		vertices[i].position.y = shapes.at(0).mesh.positions.at(j+1);
		vertices[i].position.z = shapes.at(0).mesh.positions.at(j+2);
		vertices[i].texture.x = shapes.at(0).mesh.texcoords.at(k);
		vertices[i].texture.y = shapes.at(0).mesh.texcoords.at(k + 1);
		vertices[i].normal.x = shapes.at(0).mesh.normals.at(j);
		vertices[i].normal.y = shapes.at(0).mesh.normals.at(j + 1);
		vertices[i].normal.z = shapes.at(0).mesh.normals.at(j + 2);
	}

	indices = new unsigned long[m_indexCount];
	if (!indices)
	{
		return false;
	}
	int i = 0;
	for (std::vector<unsigned int>::iterator it = shapes.at(0).mesh.indices.begin(); it != shapes.at(0).mesh.indices.end(); ++it, i++)
	{
		indices[i] = (*it);
	}

	VertexIndex* toRet = new VertexIndex;
	toRet->vertexArrayPtr = vertices;
	toRet->indexArrayPtr = indices;

	myGeometry = toRet;

	UpdateGeometry();

	return toRet;
}

void Model3D::UpdateGeometry()
{
	if (myGeometry == nullptr) return;

	D3DXMATRIX rotationMatrix;
	D3DXVECTOR3 tempPos;

	float pitch = rotation.x * 0.0174532925f;
	float yaw = rotation.y * 0.0174532925f;
	float roll = rotation.z * 0.0174532925f;

	D3DXMatrixRotationYawPitchRoll(&rotationMatrix, yaw, pitch, roll);

	for (int i = 0; i < m_vertexCount; i++)
	{
		tempPos = myGeometry->vertexArrayPtr[i].position;
		tempPos.x = tempPos.x*scale.x;
		tempPos.y = tempPos.y*scale.y;
		tempPos.z = tempPos.z*scale.z;
		D3DXVec3TransformCoord(&tempPos, &tempPos, &rotationMatrix);
		tempPos += position;
		myGeometry->vertexArrayPtr[i].position = tempPos;
	}
}
