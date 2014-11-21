#pragma once
#include "Model.h"
#include <cstring>
using namespace std;

class Model3D :
	public Model
{
private:
	VertexIndex* LoadGeometry(bool ind);
	string LoadFromFile(string filePath);
public:
	Model3D();
	Model3D(D3DXVECTOR3 position, D3DXVECTOR3 rotation, D3DXVECTOR3 scale, D3D11_USAGE usage, string filePath);
	Model3D(const Model3D&);
	~Model3D();
};

