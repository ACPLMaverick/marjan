#include "Collider.h"

Collider::Collider(SimObject* obj) : Component(obj)
{
}


Collider::~Collider()
{
}



unsigned int Collider::Initialize()
{
	PhysicsManager::GetInstance()->AddCollider(this);

	return CS_ERR_NONE;
}

unsigned int Collider::Shutdown()
{
	PhysicsManager::GetInstance()->RemoveCollider(this);
	m_collisionsSolvedWith.clear();

	return CS_ERR_NONE;
}



ColliderType Collider::GetType()
{
	return m_type;
}



bool Collider::HasAlreadyCollidedWith(Collider* col)
{
	for (std::vector<Collider*>::iterator it = m_collisionsSolvedWith.begin(); it != m_collisionsSolvedWith.end(); ++it)
	{
		if ((*it) == col)
		{
			return true;
		}	
	}

	return false;
}