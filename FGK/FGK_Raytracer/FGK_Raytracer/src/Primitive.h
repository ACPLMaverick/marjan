#pragma once

#include "stdafx.h"

class Primitive
{
protected:

#pragma region Protected

	uint64_t m_id;
	std::string m_name;

#pragma endregion

public:

#pragma region Functions Public

	Primitive(uint64_t id, const std::string* name);
	~Primitive();

	void Update();

	void Intersect();

#pragma region Accessors

	uint64_t GetUID() { return m_id; }
	const std::string* GetName() { return &m_name; }

#pragma endregion

#pragma endregion
};

