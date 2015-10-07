#include "PhysicsManager.h"


PhysicsManager::PhysicsManager()
{
}


PhysicsManager::~PhysicsManager()
{
}



unsigned int PhysicsManager::Initialize()
{
	// HARD CODED VALUES
	m_gravity = 10.0f;
	m_ifDrawColliders = true;
	///////////////////////////

	return CS_ERR_NONE;
}

unsigned int PhysicsManager::Shutdown()
{
	return CS_ERR_NONE;
}

unsigned int PhysicsManager::Run()
{
	// clear checked collider data
	for (std::vector<Collider*>::iterator it = m_colliders.begin(); it != m_colliders.end(); ++it)
	{
		(*it)->m_collisionsSolvedWith.clear();
	}

	// solve collisions (every collider with every collider, using m_collisionsSolvedWith
	for (std::vector<Collider*>::iterator it = m_colliders.begin(); it != m_colliders.end(); ++it)
	{
		for (std::vector<Collider*>::iterator it2 = m_colliders.begin(); it2 != m_colliders.end(); ++it2)
		{
			// check conditions under we do not have to check for collisions.
			if (
				*it == *it2 || 
				(*it)->HasAlreadyCollidedWith(*it2)
				)
			{
				continue;
			}

			// we can check for collision now.
			CollisonTestResult res;

			if ((*it2)->m_type == BOX_AA)
			{
				res = (*it)->TestWithBoxAA((BoxAACollider*)(*it2));
			}
			else if ((*it2)->m_type == SPHERE)
			{
				res = (*it)->TestWithSphere((SphereCollider*)(*it2));
			}

			// transform collided objects accordingly, if necessary
			if (res.ifCollision)
			{
				// only it object has moved
				if ((*it)->GetMySimObject()->GetTransform()->HasMovedLastFrame() && !(*it2)->GetMySimObject()->GetTransform()->HasMovedLastFrameFast())
				{
					(*it)->GetMySimObject()->GetTransform()->SetPosition(&((*(*it)->GetMySimObject()->GetTransform()->GetPosition()) + res.colVector));
				}
				// only it2 object has moved
				else if (!(*it)->GetMySimObject()->GetTransform()->HasMovedLastFrame() && (*it2)->GetMySimObject()->GetTransform()->HasMovedLastFrameFast())
				{
					(*it2)->GetMySimObject()->GetTransform()->SetPosition(&((*(*it2)->GetMySimObject()->GetTransform()->GetPosition()) - res.colVector));
				}
				// both objects have moved or none had
				else
				{
					res.colVector = res.colVector / 2.0f;
					glm::vec3 second = -res.colVector;

					(*it)->GetMySimObject()->GetTransform()->SetPosition(&((*(*it)->GetMySimObject()->GetTransform()->GetPosition()) + res.colVector));
					(*it2)->GetMySimObject()->GetTransform()->SetPosition(&((*(*it2)->GetMySimObject()->GetTransform()->GetPosition()) + second));
				}
			}

			// update m_collisionsSolvedWith
			(*it)->m_collisionsSolvedWith.push_back(*it2);
			(*it2)->m_collisionsSolvedWith.push_back(*it);
		}
	}

	return CS_ERR_NONE;
}



void PhysicsManager::AddCollider(Collider* col)
{
	m_colliders.push_back(col);
}

bool PhysicsManager::RemoveCollider(Collider* col)
{
	if (m_colliders.size() > 0)
	{
		for (std::vector<Collider*>::iterator it = m_colliders.begin(); it != m_colliders.end(); ++it)
		{
			if (*it == col)
			{
				m_colliders.erase(it);
				return true;
			}
		}
	}
	

	return false;
}



float PhysicsManager::GetGravity()
{
	return m_gravity;
}

bool PhysicsManager::GetIfDrawColliders()
{
	return m_ifDrawColliders;
}