#include "pch.h"
#include "MeshGLRectButton.h"


MeshGLRectButton::MeshGLRectButton(SimObject* obj, GUIButton* btn, glm::vec4* col) : MeshGLRect(obj, col)
{
	m_btn = btn;
}

MeshGLRectButton::MeshGLRectButton(const MeshGLRectButton * c) : MeshGLRect(c)
{
	m_btn = c->m_btn;
}


MeshGLRectButton::~MeshGLRectButton()
{
}



unsigned int MeshGLRectButton::Initialize()
{
	unsigned int err = MeshGLRect::Initialize();
	if (err != CS_ERR_NONE)
		return err;

	std::string font = "Font";
	m_fontShaderID = ResourceManager::GetInstance()->GetShader(&font);

	return err;
}

unsigned int MeshGLRectButton::Draw()
{
	//if (Renderer::GetInstance()->GetDrawMode() == BASIC)
	ShaderID* ids = Renderer::GetInstance()->GetCurrentShaderID();
	if(ids != m_fontShaderID)
		Renderer::GetInstance()->SetCurrentShader(m_fontShaderID);

	glm::mat4 wvp;
	if (m_btn == nullptr)
	{
		wvp = ((*System::GetInstance()->GetCurrentScene()->GetCamera()->GetViewProjMatrix()) *
			(*m_obj->GetTransform()->GetWorldMatrix()));
	}
	else
	{
		wvp = *m_btn->GetTransformMatrix();
	}

	glUniformMatrix4fv(m_fontShaderID->id_worldViewProj, 1, GL_FALSE, &(wvp)[0][0]);

	// here we will set up texture?
	if (m_texID != nullptr)
		glBindTexture(GL_TEXTURE_2D, m_texID->id);
	else
		glBindTexture(GL_TEXTURE_2D, NULL);

	//////////////////////////////////////////

	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);
	glEnableVertexAttribArray(2);
	glEnableVertexAttribArray(3);

	glBindBuffer(GL_ARRAY_BUFFER, m_vertexData->ids->vertexBuffer);
	glVertexAttribPointer(
		0,
		3,
		GL_FLOAT,
		GL_FALSE,
		0,
		(void*)0
		);

	glBindBuffer(GL_ARRAY_BUFFER, m_vertexData->ids->uvBuffer);
	glVertexAttribPointer(
		1,
		2,
		GL_FLOAT,
		GL_FALSE,
		0,
		(void*)0
		);

	glBindBuffer(GL_ARRAY_BUFFER, m_vertexData->ids->normalBuffer);
	glVertexAttribPointer(
		2,
		3,
		GL_FLOAT,
		GL_FALSE,
		0,
		(void*)0
		);

	glBindBuffer(GL_ARRAY_BUFFER, m_vertexData->ids->colorBuffer);
	glVertexAttribPointer(
		3,
		4,
		GL_FLOAT,
		GL_FALSE,
		0,
		(void*)0
		);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_vertexData->ids->indexBuffer);


	glDrawElements(
		GL_TRIANGLES,
		m_vertexData->data->indexCount,
		GL_UNSIGNED_INT,
		(void*)0
		);


	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);
	glDisableVertexAttribArray(2);
	glDisableVertexAttribArray(3);

	if (ids != m_fontShaderID)
		Renderer::GetInstance()->SetCurrentShader(ids);

	return CS_ERR_NONE;
}
