#pragma once
#include "Model.h"
class Sprite2D :
	public Model
{
protected:
	virtual VertexIndex* LoadGeometry(bool ind);
public:
	Sprite2D();
	Sprite2D(D3DXVECTOR3 position, D3DXVECTOR3 rotation, D3DXVECTOR3 scale, D3D11_USAGE usage);
	~Sprite2D();
};

