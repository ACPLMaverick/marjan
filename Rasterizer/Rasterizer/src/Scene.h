#pragma once

#include "Buffer.h"

#include <vector>

class Primitive;
class Camera;

class Scene
{
protected:

#pragma region Elements

	std::vector<Primitive*> _primitives;
	std::vector<Camera*> _cameras;

#pragma endregion

#pragma region Variables

	std::string _name = "NoNameScene";
	uint32_t _uID = (uint32_t)-1;
	uint32_t _currentCamera = 0;

	std::vector<Primitive*> _primitivesToAdd;
	std::vector<std::vector<Primitive*>::iterator> _primitivesToRemove;
	bool _flagToAddPrimitive = false;
	bool _flagToRemovePrimitive = false;

#pragma endregion

#pragma region MethodsInternal

	virtual void InitializeScene() = 0;

#pragma endregion
public:
	Scene();
	~Scene();

#pragma region MethodsMain

	void Initialize(uint32_t uID, std::string* name);
	void Shutdown();
	void Update();
	void Draw();

#pragma endregion

#pragma region Accessors

	uint32_t GetUID() { return _uID; }
	const std::string* GetName() { return &_name; }
	Camera* const GetCurrentCamera() { return (_cameras.size() > 0 ? _cameras[_currentCamera] : nullptr); }

	Camera* const GetCamera(uint32_t uid);
	Camera* const GetCamera(std::string* name);

	void AddPrimitive(Primitive* const Primitive);
	void AddCamera(Camera* const camera);

	Camera* const RemoveCamera(uint32_t uid);
	Camera* const RemoveCamera(std::string* name);

#pragma endregion
};
