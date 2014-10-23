#pragma once
#include "Model.h"
class Sprite2D :
	public Model
{
protected:
	virtual VertexIndex* LoadGeometry();
public:
	Sprite2D();
	Sprite2D(D3DXVECTOR3 position, D3DXVECTOR3 rotation, D3DXVECTOR3 scale);
	~Sprite2D();
};

