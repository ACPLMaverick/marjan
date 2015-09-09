#pragma once

/*
	It governs loading, keeping in memory and freeing any external resources the simulation may use.
*/

#include "Common.h"
#include "Singleton.h"
#include "Texture.h"

#include <string>
#include <map>

struct TextureID
{
	int id;
	std::string name;
};

struct ShaderID
{
	int id;
	std::string name;

	int id_worldViewProj;
	int id_world;
	int id_worldInvTrans;
	int id_eyeVector;
	int id_lightDir;
	int id_lightDiff;
	int id_lightSpec;
	int id_lightAmb;
	int id_gloss;
	int id_highlight;
};

class ResourceManager : public Singleton<ResourceManager>
{
	friend class Singleton<ResourceManager>;

private:
	std::map<std::string, TextureID*> m_texturesNames;
	std::map<std::string, ShaderID*> m_shadersNames;

	std::map<unsigned int, TextureID*> m_texturesIds;
	std::map<unsigned int, ShaderID*> m_shadersIds;

	ResourceManager();
	TextureID* LoadTextureData(const std::string*);
	ShaderID* LoadShaderData(const std::string*, const std::string*);
public:
	ResourceManager(const ResourceManager*);
	~ResourceManager();

	unsigned int Initialize();
	unsigned int Shutdown();

	TextureID* LoadTexture(const std::string*);
	ShaderID* LoadShader(const std::string*);
	ShaderID* LoadShader(const std::string*, const std::string*);

	TextureID* GetTexture(const std::string*);
	TextureID* GetTexture(unsigned int);
	ShaderID* GetShader(const std::string*);
	ShaderID* GetShader(unsigned int);
};

