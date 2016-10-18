#include "Camera.h"



Camera::Camera(
	const math::Float3* pos,
	const math::Float3* tgt,
	const math::Float3* up,
	float fovY,
	float ar,
	float np,
	float fp
) :
	_position(*pos),
	_target(*tgt),
	_up(*up),
	_fovY(fovY),
	_aspectRatio(ar),
	_nearPlane(np),
	_farPlane(fp)
{
	math::Matrix4x4::LookAt(&_position, &_target, &_up, &_viewMatrix);
	math::Matrix4x4::Perspective(_fovY, _aspectRatio, _nearPlane, _farPlane, &_projMatrix);
	_viewProjMatrix = _viewMatrix * _projMatrix;

	_direction = _target - _position;
	math::Float3::Normalize(_direction);
}


Camera::~Camera()
{
}

void Camera::Update()
{
	if (_viewMatrixNeedUpdate || _projMatrixNeedUpdate)
	{
		if (_viewMatrixNeedUpdate)
		{
			math::Matrix4x4::LookAt(&_position, &_target, &_up, &_viewMatrix);
			_direction = _target - _position;
			math::Float3::Normalize(_direction);
			_viewMatrixNeedUpdate = false;
		}
		if (_projMatrixNeedUpdate)
		{
			math::Matrix4x4::Perspective(_fovY, _aspectRatio, _nearPlane, _farPlane, &_projMatrix);
			_projMatrixNeedUpdate = false;
		}
		_viewProjMatrix = _viewMatrix * _projMatrix;
	}
}