#include "ResourceManager.h"
#include "Renderer.h"


ResourceManager::ResourceManager()
{
}

ResourceManager::ResourceManager(const ResourceManager*)
{
}

ResourceManager::~ResourceManager()
{
}



unsigned int ResourceManager::Initialize()
{
	////////////// HARD-CODED INITIALIZATION GOES HERE

	// Textures

	return CS_ERR_NONE;
}

unsigned int ResourceManager::Shutdown()
{
	for (std::map<std::string, TextureID*>::iterator it = m_texturesNames.begin(); it != m_texturesNames.end(); ++it)
	{
		// shutdown
		delete it->second;
	}

	for (std::map<std::string, ShaderID*>::iterator it = m_shadersNames.begin(); it != m_shadersNames.end(); ++it)
	{
		// shutdown
		delete it->second;
	}

	return CS_ERR_NONE;
}



TextureID* ResourceManager::LoadTexture(const std::string* path)
{
	if (m_texturesNames.find(*path) != m_texturesNames.end())
	{
		return m_texturesNames[*path];
	}
	else return LoadTextureData(path);
}

ShaderID* ResourceManager::LoadShader(const std::string* name)
{
	if (m_shadersNames.find(*name) != m_shadersNames.end())
	{
		return m_shadersNames[*name];
	}
	else return LoadShaderData(name, name);
}

ShaderID* ResourceManager::LoadShader(const std::string* name, const std::string* nameFrag)
{
	if (m_shadersNames.find(*nameFrag) != m_shadersNames.end())
	{
		return m_shadersNames[*nameFrag];
	}
	else return LoadShaderData(name, nameFrag);
}



TextureID* ResourceManager::GetTexture(const std::string* path)
{
	if (m_texturesNames.find(*path) != m_texturesNames.end())
	{
		return m_texturesNames[*path];
	}
	else return nullptr;
}

TextureID* ResourceManager::GetTexture(unsigned int id)
{
	if (m_texturesIds.find(id) != m_texturesIds.end())
	{
		return m_texturesIds[id];
	}
	else return nullptr;
}

ShaderID* ResourceManager::GetShader(const std::string* name)
{
	if (m_shadersNames.find(*name) != m_shadersNames.end())
	{
		return m_shadersNames[*name];
	}
	else return nullptr;
}

ShaderID* ResourceManager::GetShader(unsigned int id)
{
	if (m_shadersIds.find(id) != m_shadersIds.end())
	{
		return m_shadersIds[id];
	}
	else return nullptr;
}



TextureID* ResourceManager::LoadTextureData(const std::string* path)
{
	TextureID* ntex = new TextureID;
	ntex->id = -1;

	Renderer::LoadTexture(path, ntex);

	if (ntex->id == -1)
	{
		delete ntex;
		return nullptr;
	}

	m_texturesNames.emplace(ntex->name, ntex);
	m_texturesIds.emplace(ntex->id, ntex);

	return ntex;
}

ShaderID* ResourceManager::LoadShaderData(const std::string* nameVert, const std::string* nameFrag)
{
	std::string vertexName = *nameVert + "VertexShader";
	std::string fragmentName = *nameFrag + "FragmentShader";

	ShaderID* newShader = new ShaderID;
	newShader->id = -1;

	Renderer::LoadShaders(&vertexName, &fragmentName, nameFrag, newShader);

	if (newShader->id == -1)
	{
		delete newShader;
		return nullptr;
	}

	m_shadersNames.emplace(newShader->name, newShader);
	m_shadersIds.emplace(newShader->id, newShader);

	return newShader;
}