#include "Ray.h"

Ray::Ray()
{
	this->_origin = math::Float3();
	this->_dir = math::Float3();
	this->_distance = 0.0f;
}

Ray::Ray(math::Float3& s, math::Float3& d)
{
	this->_origin = s;
	this->_dir = d;
}

math::Float3 Ray::GetOrigin()
{
	return _origin;
}

math::Float3 Ray::GetDirection()
{
	return _dir;
}