#pragma once

#include "stdafx.h"

class Camera
{
protected:

#pragma region Protected

	uint64_t _id;
	std::string _name;

#pragma endregion

public:

#pragma region Functions Public

	Camera();
	~Camera();

	void Update();

#pragma region Accessors

	uint64_t GetUID() { return _id; }
	const std::string* GetName() { return &_name; }

#pragma endregion

#pragma endregion
};

