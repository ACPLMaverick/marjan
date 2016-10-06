#include "TriangleMav.h"
#include "RendererMav.h"
#include "GraphicsDevice.h"
#include "../System.h"
#include "../Camera.h"
#include "../Scene.h"

namespace rendererMav
{
	TriangleMav::TriangleMav(math::Float3 x, math::Float3 y, math::Float3 z,
		math::Float3 ux, math::Float3 uy, math::Float3 uz,
		math::Float3 cx, math::Float3 cy, math::Float3 cz,
		Color32 col) :
		Triangle(x, y, z, ux, uy, uz, cx, cy, cz, col)
	{
	}

	TriangleMav::~TriangleMav()
	{
	}

	void TriangleMav::Update()
	{
		math::Float3 trans = *_transform.GetRotation();
		trans.x += 0.1f;
		_transform.SetRotation(&trans);
	}

	void TriangleMav::Draw()
	{
		GraphicsDevice* gd = ((RendererMav*)System::GetInstance()->GetRenderer())->GetGraphicsDevice();
		gd->SetVertexBuffer(&v1);
		gd->SetUVBuffer(&u1);
		gd->SetColorBuffer(&c1);

		Camera* cam = System::GetInstance()->GetCurrentScene()->GetCurrentCamera();
		math::Matrix4x4 wvp = *_transform.GetWorldMatrix() * *cam->GetViewProjMatrix();
		gd->SetWorldViewProjMatrix(&wvp);

		gd->Draw(1);
	}
}