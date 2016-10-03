#include "Ray.h"

Ray::Ray()
{
	this->_start = math::Float3();
	this->_dir = math::Float3(0, -1.0f, 0);
}

Ray::Ray(math::Float3& s, math::Float3& d)
{
	this->_start = s;
	this->_dir = d;
}