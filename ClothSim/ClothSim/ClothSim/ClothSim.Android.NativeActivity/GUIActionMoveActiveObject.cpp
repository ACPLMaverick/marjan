#include "pch.h"
#include "GUIActionMoveActiveObject.h"


GUIActionMoveActiveObject::GUIActionMoveActiveObject(GUIButton* b) : GUIAction(b)
{
}

GUIActionMoveActiveObject::GUIActionMoveActiveObject(const GUIActionMoveActiveObject * c) : GUIAction(c)
{
}


GUIActionMoveActiveObject::~GUIActionMoveActiveObject()
{
}

unsigned int GUIActionMoveActiveObject::Action(std::vector<void*>* params, const glm::vec2* clickPos)
{
	unsigned int err = CS_ERR_NONE;

	if (params->size() != 1)
		return CS_ERR_ACTION_BAD_PARAM;

	SimObject* cObj = System::GetInstance()->GetCurrentScene()->GetObject();

	glm::vec3 mVector = glm::vec3();
	MovementDirection dir = (MovementDirection)(int)(params->at(0));
	float scl = cObj->GetTransform()->GetScale()->y;
	float pos = cObj->GetTransform()->GetPosition()->y;
	switch (dir)
	{
	case GUIActionMoveActiveObject::FORWARD:
		mVector = glm::vec3(0.0f, 0.0f, -1.0f);
		break;
	case GUIActionMoveActiveObject::BACKWARD:
		mVector = glm::vec3(0.0f, 0.0f, 1.0f);
		break;
	case GUIActionMoveActiveObject::LEFT:
		mVector = glm::vec3(-1.0f, 0.0f, 0.0f);
		break;
	case GUIActionMoveActiveObject::RIGHT:
		mVector = glm::vec3(1.0f, 0.0f, 0.0f);
		break;
	case GUIActionMoveActiveObject::UP:
		mVector = glm::vec3(0.0f, 1.0f, 0.0f);
		break;
	case GUIActionMoveActiveObject::DOWN:
		if((pos - scl) > System::GetInstance()->GetCurrentScene()->GetGroundLevel())
			mVector = glm::vec3(0.0f, -1.0f, 0.0f);
		break;
	default:
		mVector = glm::vec3();
		break;
	}

	glm::vec3 cPosVector = cObj->GetTransform()->GetPositionCopy();

	mVector = mVector * BOX_SPEED * (float)Timer::GetInstance()->GetDeltaTime();

	glm::vec3 addedVector = cPosVector + mVector;
	cObj->GetTransform()->SetPosition(&addedVector);

	return err;
}

