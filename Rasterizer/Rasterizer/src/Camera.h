#pragma once

#include "stdafx.h"
#include "Matrix4x4.h"
#include "Float3.h"

class Camera
{
protected:

#pragma region Protected

	math::Matrix4x4 _viewProjMatrix;
	math::Matrix4x4 _viewMatrix;
	math::Matrix4x4 _projMatrix;

	math::Float3 _position;
	math::Float3 _target;
	math::Float3 _up;

	math::Float3 _direction;

	uint64_t _id;
	std::string _name;

	float _fovY;
	float _aspectRatio;
	float _nearPlane;
	float _farPlane;

	bool _viewMatrixNeedUpdate = false;
	bool _projMatrixNeedUpdate = false;

#pragma endregion

#pragma region Functions Protected

#pragma endregion

public:

#pragma region Functions Public

	Camera(
		const math::Float3* pos,
		const math::Float3* tgt,
		const math::Float3* up,
		float fovY,
		float ar,
		float np = 0.01f,
		float fp = 100.0f
	);
	~Camera();

	void Update();

#pragma region Accessors

	uint64_t GetUID() { return _id; }
	const std::string* GetName() { return &_name; }

	const math::Matrix4x4* GetViewProjMatrix() const { return &_viewProjMatrix; }
	const math::Matrix4x4* GetViewMatrix() const { return &_viewMatrix; }
	const math::Matrix4x4* GetProjMatrix() const { return &_projMatrix; }

	const math::Float3* GetPosition() const { return &_position; }
	const math::Float3* GetTarget() const { return &_target; }
	const math::Float3* GetUp() const { return &_up; }
	const math::Float3* GetDirection() const { return &_direction; }

	void SetPosition(const math::Float3* pos) { _position = *pos; _viewMatrixNeedUpdate = true; }
	void SetTarget(const math::Float3* tgt) { _target = *tgt; _viewMatrixNeedUpdate = true; }
	void SetUp(const math::Float3* up) { _up = *up; _viewMatrixNeedUpdate = true; }

#pragma endregion

#pragma endregion
};

