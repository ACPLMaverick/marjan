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

	m_vertexCount = 0;
	m_indexCount = 0;
	for (int f = 0; f < shapes.size(); f++)
	{
		m_vertexCount += (shapes.at(f).mesh.positions.size() / 3);
		m_indexCount += shapes.at(f).mesh.indices.size();
	}

	vertices = new Vertex[m_vertexCount];
	if (!vertices)
	{
		return false;
	}

	indices = new unsigned long[m_indexCount];
	if (!indices)
	{
		return false;
	}
	int i = 0;
	int p = 0;
	for (int w = 0; w < shapes.size(); w++)
	{
		std::vector<float>::iterator itp = shapes.at(w).mesh.positions.begin();
		std::vector<float>::iterator itt = shapes.at(w).mesh.texcoords.begin();
		std::vector<float>::iterator itn = shapes.at(w).mesh.normals.begin();
		std::vector<unsigned int>::reverse_iterator iti = shapes.at(w).mesh.indices.rbegin();
		for (; itp != shapes.at(w).mesh.positions.end(); itp+=3, itt+=2, itn+=3, i++)
		{
			vertices[i].position.x =  - (*itp);
			vertices[i].position.y =  (*(itp+1));
			vertices[i].position.z =  (*(itp+2));

			vertices[i].texture.x = (*itt);
			vertices[i].texture.y = 1.0f - (*(itt + 1));

			vertices[i].normal.x = (*itn);
			vertices[i].normal.y = (*(itn + 1));
			vertices[i].normal.z = (*(itn + 2));
		}

		for (; iti != shapes.at(w).mesh.indices.rend(); ++iti, p++)
		{
			indices[p] = (*iti);
		}
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
