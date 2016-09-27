#pragma once

// a container for all game elements, also govern their update and draw cycle
// This is an abstract class. InitializeScene can be implemented to set
// objects which will be contained here

#include <vector>

class Primitive;
class Camera;

class Scene
{
protected:

#pragma region Elements

	std::vector<Primitive*> m_primitives;
	std::vector<Camera*> m_cameras;

#pragma endregion

#pragma region Variables

	std::string m_name = "NoNameScene";
	uint32_t m_uID = (uint32_t)-1;
	uint32_t m_currentCamera = 0;

	std::vector<Primitive*> m_primitivesToAdd;
	std::vector<std::vector<Primitive*>::iterator> m_primitivesToRemove;
	bool m_flagToAddPrimitive = false;
	bool m_flagToRemovePrimitive = false;

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

#pragma endregion

#pragma region Accessors

	uint32_t GetUID() { return m_uID; }
	const std::string* GetName() { return &m_name; }
	Camera* const GetCurrentCamera() { return (m_cameras.size() > 0 ? m_cameras[m_currentCamera] : nullptr); }

	Primitive* const GetPrimitive(uint32_t uid);
	Primitive* const GetPrimitive(std::string* name);
	Camera* const GetCamera(uint32_t uid);
	Camera* const GetCamera(std::string* name);

	void AddPrimitive(Primitive* const Primitive);
	void AddCamera(Camera* const camera);

	Primitive* const RemovePrimitive(uint32_t uid);
	Primitive* const RemovePrimitive(std::string* name);
	Camera* const RemoveCamera(uint32_t uid);
	Camera* const RemoveCamera(std::string* name);

#pragma endregion
};

