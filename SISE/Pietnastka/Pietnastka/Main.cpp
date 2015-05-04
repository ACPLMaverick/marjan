#include "Header.h"

using namespace std;

//////////////////////////////////////////////
////// GLOBALS

vector<Node*> globalNodeList;

class node_priority_queue : public priority_queue<Node*, vector<Node*>, NodeComparator> 
{
public:
	bool contains(Node* node)
	{
		for (vector<Node*>::iterator it = this->c.begin(); it != this->c.end(); ++it)
		{
			if ((*node) == (*(*it)))
				return true;
		}
		return false;
	}

	bool containsSameDistanceLowerCost(Node* node)
	{
		for (vector<Node*>::iterator it = this->c.begin(); it != this->c.end(); ++it)
		{
			if (node->distance == (*it)->distance && node->cost >= (*it)->cost)
				return true;
		}
		return false;
	}
};

//////////////////////////////////////////////

inline cost_t Abs(cost_t a)
{
	return (a) > 0 ? (a) : (-a);
}

void PrintNode(const Node* state)
{
	for (int i = 0; i < RIDDLE_SIZE; ++i)
	{
		for (int j = 0; j < RIDDLE_SIZE; ++j)
		{
			cout.width(3);

			if (state->board[i][j] != CURRENT_FIELD)
				cout << (int)state->board[i][j] << " ";
			else
				cout << state->board[i][j] << " ";
		}
		cout << endl;
	}
	cout << endl;
}

Node* GenerateStartNode()
{
	Node* state = new Node;
	int ctr = 1;
	for (int i = 0; i < RIDDLE_SIZE; ++i)
	{
		for (int j = 0; j < RIDDLE_SIZE; ++j)
		{
			state->board[i][j] = ctr;
			++ctr;
		}
	}
	state->board[RIDDLE_SIZE - 1][RIDDLE_SIZE - 1] = CURRENT_FIELD;
	state->currentX = RIDDLE_SIZE - 1;
	state->currentY = RIDDLE_SIZE - 1;
	state->ifMarked = false;
	state->parent = NULL;

	globalNodeList.push_back(state);

	return state;
}

void ShuffleNode(Node* state, const unsigned int level)
{
	srand(time(NULL));
	unsigned char move;
	unsigned char temp;
	unsigned char lastMove = 4;

	for (int i = 0; i < level; /**/)
	{
		move = rand() % 4;

		switch (move)
		{
		case MOVE_UP:
			if (state->currentY < 1 || lastMove == MOVE_DOWN)
				continue;
			temp = state->board[state->currentY - 1][state->currentX];
			state->board[state->currentY - 1][state->currentX] = state->board[state->currentY][state->currentX];
			state->board[state->currentY][state->currentX] = temp;

			state->currentY -= 1;
			break;

		case MOVE_DOWN:
			if (state->currentY > RIDDLE_SIZE - 2 || lastMove == MOVE_UP)
				continue;
			temp = state->board[state->currentY + 1][state->currentX];
			state->board[state->currentY + 1][state->currentX] = state->board[state->currentY][state->currentX];
			state->board[state->currentY][state->currentX] = temp;

			state->currentY += 1;
			break;

		case MOVE_LEFT:
			if (state->currentX < 1 || lastMove == MOVE_RIGHT)
				continue;
			temp = state->board[state->currentY][state->currentX - 1];
			state->board[state->currentY][state->currentX - 1] = state->board[state->currentY][state->currentX];
			state->board[state->currentY][state->currentX] = temp;

			state->currentX -= 1;
			break;

		case MOVE_RIGHT:
			if (state->currentX > RIDDLE_SIZE - 2 || lastMove == MOVE_LEFT)
				continue;
			temp = state->board[state->currentY][state->currentX + 1];
			state->board[state->currentY][state->currentX + 1] = state->board[state->currentY][state->currentX];
			state->board[state->currentY][state->currentX] = temp;

			state->currentX += 1;
			break;

		default:
			cout << "ERROR: Wrong number given by rand()" << endl;
			break;
		}

		lastMove = move;
		++i;
	}
}

cost_t GenerateCostManhattan(const Node* node)
{
	cost_t cost = 0.0f;
	cost_t tempCost;
	unsigned char endX, endY;

	for (int i = 0; i < RIDDLE_SIZE; ++i)
	{
		for (int j = 0; j < RIDDLE_SIZE; ++j)
		{
			if (node->board[i][j] != CURRENT_FIELD)
			{
				endX = (node->board[i][j] - 1) % 4;
				endY = (node->board[i][j] - 1) / 4;
			}
			else
			{
				endX = 3;
				endY = 3;
			}

			tempCost = (Abs(j - endX) + Abs(i - endY));
			cost += tempCost;
		}
	}

	return cost;
}

bool CreateNodeFromMove(Node* oldState, Node* newState, unsigned int dir)
{
	(*newState) = (*oldState);
	newState->ifMarked = false;
	newState->parent = oldState;
	unsigned char temp;

	switch (dir)
	{
	case MOVE_UP:
		if (newState->currentY < 1)
			return false;
		temp = newState->board[newState->currentY - 1][newState->currentX];
		newState->board[newState->currentY - 1][newState->currentX] = newState->board[newState->currentY][newState->currentX];
		newState->board[newState->currentY][newState->currentX] = temp;

		newState->currentY -= 1;
		break;

	case MOVE_DOWN:
		if (newState->currentY > RIDDLE_SIZE - 2)
			return false;
		temp = newState->board[newState->currentY + 1][newState->currentX];
		newState->board[newState->currentY + 1][newState->currentX] = newState->board[newState->currentY][newState->currentX];
		newState->board[newState->currentY][newState->currentX] = temp;

		newState->currentY += 1;
		break;

	case MOVE_LEFT:
		if (newState->currentX < 1)
			return false;
		temp = newState->board[newState->currentY][newState->currentX - 1];
		newState->board[newState->currentY][newState->currentX - 1] = newState->board[newState->currentY][newState->currentX];
		newState->board[newState->currentY][newState->currentX] = temp;

		newState->currentX -= 1;
		break;

	case MOVE_RIGHT:
		if (newState->currentX > RIDDLE_SIZE - 2)
			return false;
		temp = newState->board[newState->currentY][newState->currentX + 1];
		newState->board[newState->currentY][newState->currentX + 1] = newState->board[newState->currentY][newState->currentX];
		newState->board[newState->currentY][newState->currentX] = temp;

		newState->currentX += 1;
		break;

	default:
		cout << "ERROR: Wrong number given to CreateStateFromMove()" << endl;
		return false;
		break;
	}

	if (oldState->cost < FLT_MAX)
		newState->cost = (cost_t)(oldState->distance + 1) + GenerateCostManhattan(newState);

	return true;
}

Node* CreateNode()
{
	Node* state = new Node;

	globalNodeList.push_back(state);

	return state;
}

inline void FillReturnStack(Node* current, stack<Node*>* returnStack)
{
	Node* retNode = current;
	do
	{
		returnStack->push(retNode);
		retNode = retNode->parent;
	} while (retNode != NULL);
}

bool CheckIfNodeMatchesStartNode(const Node* state)
{
	if (state->currentX != RIDDLE_SIZE - 1 || state->currentY != RIDDLE_SIZE - 1)
		return false;

	int ctr = 1;
	for (int i = 0; i < RIDDLE_SIZE; ++i)
	{
		for (int j = 0; j < RIDDLE_SIZE; ++j)
		{
			if (!(i == RIDDLE_SIZE - 1 && j == RIDDLE_SIZE - 1))
			{
				if (state->board[i][j] != ctr)
					return false;

				++ctr;
			}
		}
	}
	return true;
}

//////////////////////////////////////////////

inline void GenerateAdjacentStates(Node* node)
{
	for (int i = 0; i < 4; ++i)
	{
		Node* dir = new Node;

		if (!CreateNodeFromMove(node, dir, 3 - i))
		{
			delete dir;
			node->neighbours[i] = NULL;
		}
		else if ((*dir) == (*(dir->parent)))
		{
			node->neighbours[i] = dir->parent;
			delete dir;
		}
		else
		{
			bool assigned = false;
			for (vector<Node*>::iterator it = globalNodeList.begin(); it != globalNodeList.end(); ++it)
			{
				if ((*dir) == (*(*it)))
				{
					node->neighbours[i] = (*it);
					delete dir;
					assigned = true;
					break;
				}
			}

			if (!assigned)
			{
				globalNodeList.push_back(dir);
				node->neighbours[i] = dir;
			}
		}
	}
}

bool DFS(Node* node, stack<Node*>* returnStack)
{
	node->ifMarked = true;
	returnStack->push(node);

	if (CheckIfNodeMatchesStartNode(node))
	{
		return true;
	}
	else
	{
		GenerateAdjacentStates(node);

		// traverse every one of them
		for (int i = 0; i < 4; ++i)
		{
			if (node->neighbours[i] != NULL && !(node->neighbours[i]->ifMarked))
			{
				if (DFS(node->neighbours[i], returnStack))
				{
					return true;
				}
			}
		}

		returnStack->pop();
		return false;
	}
}

bool DFSIterative(Node* node, stack<Node*>* returnStack)
{
	stack<Node*> st;
	st.push(node);

	while (!st.empty())
	{
		Node* n = st.top();
		st.pop();

		if (n != NULL && !n->ifMarked)
		{
			if (CheckIfNodeMatchesStartNode(n))
			{
				FillReturnStack(n, returnStack);
				return true;
			}

			GenerateAdjacentStates(n);

			n->ifMarked = true;

			for (int i = 0; i < 4; ++i)
			{
				st.push(n->neighbours[3 - i]);
			}
		}
	}

	return false;
}

bool BFS(Node* node, stack<Node*>* returnStack)
{
	queue<Node*> q;

	q.push(node);
	node->ifMarked = true;

	while (!q.empty())
	{
		Node* n = q.front();
		q.pop();

		GenerateAdjacentStates(n);
		for (int i = 0; i < 4; ++i)
		{
			if (n->neighbours[i] != NULL && !(n->neighbours[i]->ifMarked))
			{
				q.push(n->neighbours[i]);
				n->neighbours[i]->ifMarked = true;

				if (CheckIfNodeMatchesStartNode(n->neighbours[i]))
				{
					FillReturnStack(n->neighbours[i], returnStack);
					return true;
				}
			}
		}
	}

	return false;
}

bool AStar(Node* node, stack<Node*>* returnStack)
{
	node_priority_queue open;
	node_priority_queue closed;
	node->cost = 0;
	open.push(node);

	while (!open.empty())
	{
		Node* n = open.top();
		open.pop();

		GenerateAdjacentStates(n);

		for (int i = 0; i < 4; ++i)
		{
			if (n->neighbours[i] != NULL)
			{
				if (CheckIfNodeMatchesStartNode(n->neighbours[i]))
				{
					FillReturnStack(n->neighbours[i], returnStack);
					return true;
				}

				n->neighbours[i]->distance = n->distance + 1;
				n->neighbours[i]->cost = GenerateCostManhattan(n->neighbours[i]) + n->neighbours[i]->distance;

				if (!(open.containsSameDistanceLowerCost(n->neighbours[i]) || closed.containsSameDistanceLowerCost(n->neighbours[i])))
				{
					open.push(n->neighbours[i]);
				}
			}
		}

		closed.push(n);
	}

	return false;
}

void Search(Node* start, unsigned int method, stack<Node*>* returnStack)
{
	bool result;

	switch (method)
	{
	case 0:
		result = DFS(start, returnStack);
		break;
	case 1:
		result = DFSIterative(start, returnStack);
		break;
	case 2:
		result = BFS(start, returnStack);
		break;
	case 3:
		result = AStar(start, returnStack);
		break;
	default:
		cout << "ERROR: Wrong method ID given to Search() function.";
		break;
	}
}

//////////////////////////////////////////////

inline void TraverseAndPrintPath(stack<Node*>* returnStack)
{
	Node* ours;
	while (!returnStack->empty())
	{
		ours = returnStack->top();
		PrintNode(ours);
		returnStack->pop();
	}
}

inline void DeleteAllNodes()
{
	for (vector<Node*>::iterator it = globalNodeList.begin(); it != globalNodeList.end(); ++it)
	{
		delete (*it);
	}
	globalNodeList.clear();
}

//////////////////////////////////////////////

int main()
{
	Node* startNode = GenerateStartNode();

	ShuffleNode(startNode, 3);

	cout << "Shuffled start state: " << endl << endl;
	PrintNode(startNode);

	stack<Node*> returnStack;
	Search(startNode, 3, &returnStack);

	cout << "Solution path: " << endl << endl;
	TraverseAndPrintPath(&returnStack);

	getch();

	DeleteAllNodes();

	return 0;
}