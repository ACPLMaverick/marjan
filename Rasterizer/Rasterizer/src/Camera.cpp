#include "Camera.h"
#include "Input.h"
#include "Timer.h"

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
	math::Matrix4x4::Inverse(&_viewMatrix, &_viewInvMatrix);
	_viewProjMatrix = _viewMatrix * _projMatrix;

	_direction = _target - _position;
	math::Float3::Normalize(_direction);
}


Camera::~Camera()
{
}

void Camera::Update()
{
	// camera movement
	if (Input::GetInstance()->GetKeyDown('W'))
	{
		_position = _position + math::Float3(0.0f, 0.0f, _cameraSpeed * (float)Timer::GetInstance()->GetDeltaTime());
		_viewMatrixNeedUpdate = true;
	}
	if (Input::GetInstance()->GetKeyDown('A'))
	{
		_position = _position + math::Float3(-_cameraSpeed * (float)Timer::GetInstance()->GetDeltaTime(), 0.0f, 0.0f);
		_viewMatrixNeedUpdate = true;
	}
	if (Input::GetInstance()->GetKeyDown('S'))
	{
		_position = _position + math::Float3(0.0f, 0.0f, -_cameraSpeed * (float)Timer::GetInstance()->GetDeltaTime());
		_viewMatrixNeedUpdate = true;
	}
	if (Input::GetInstance()->GetKeyDown('D'))
	{
		_position = _position + math::Float3(_cameraSpeed * (float)Timer::GetInstance()->GetDeltaTime(), 0.0f, 0.0f);
		_viewMatrixNeedUpdate = true;
	}
	if (Input::GetInstance()->GetKeyDown('Q'))
	{
		_position = _position + math::Float3(0.0f, _cameraSpeed * (float)Timer::GetInstance()->GetDeltaTime(), 0.0f);
		_viewMatrixNeedUpdate = true;
	}
	if (Input::GetInstance()->GetKeyDown('Z'))
	{
		_position = _position + math::Float3(0.0f, -_cameraSpeed * (float)Timer::GetInstance()->GetDeltaTime(), 0.0f);
		_viewMatrixNeedUpdate = true;
	}

	// update of the matrices

	if (_viewMatrixNeedUpdate || _projMatrixNeedUpdate)
	{
		if (_viewMatrixNeedUpdate)
		{
			math::Matrix4x4::LookAt(&_position, &_target, &_up, &_viewMatrix);
			math::Matrix4x4::Inverse(&_viewMatrix, &_viewInvMatrix);
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