#include "Header.h"
#include "Tester.h"
#include <Windows.h>

using namespace std;

//////////////////////////////////////////////
////// GLOBALS

vector<Node*> globalNodeList;

double pcFreq = 0.0;
__int64 counterStart = 0;

void StartCounter()
{
	LARGE_INTEGER li;
	QueryPerformanceFrequency(&li);

	pcFreq = double(li.QuadPart) / 1000.0;

	QueryPerformanceCounter(&li);
	counterStart = li.QuadPart;
}

double GetCounter()
{
	LARGE_INTEGER li;
	QueryPerformanceCounter(&li);
	return double(li.QuadPart - counterStart) / pcFreq;
}

//////////////////////////////////////////////

inline int Abs(int a)
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
	state->parent = nullptr;

	globalNodeList.push_back(state);

	return state;
}

void ShuffleNode(Node* state, const unsigned int level)
{
	srand(time(nullptr));
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

inline cost_t GenerateCostManhattan(const Node* node)
{
	cost_t cost = 0;
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

inline bool CreateNodeFromMove(Node* oldState, Node* newState, unsigned int dir)
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

inline void FillReturnStack(Node* current, node_stack* returnStack)
{
	Node* retNode = current;
	do
	{
		returnStack->push(retNode);
		retNode = retNode->parent;
	} while (retNode != nullptr);
}

inline bool CheckIfNodeMatchesStartNode(const Node* state)
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
			node->neighbours[i] = nullptr;
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

inline void GenerateDumbAdjacentStates(Node* node)
{
	for (int i = 0; i < 4; ++i)
	{
		Node* dir = new Node;

		if (!CreateNodeFromMove(node, dir, 3 - i))
		{
			delete dir;
			node->neighbours[i] = nullptr;
		}
		else
		{
			globalNodeList.push_back(dir);
			node->neighbours[i] = dir;
		}
	}
}

inline bool CheckForExistingNode(Node* node, vector<Node*>* vec)
{
	for (vector<Node*>::iterator it = vec->begin(); it != vec->end(); ++it)
	{
		if (node == (*it))
		{
			return true;
		}
	}
	return false;
}

bool DFS(Node* node, node_stack* returnStack, unsigned long &steps, unsigned long maxSteps)
{
	node->ifMarked = true;
	returnStack->push(node);

	if (CheckIfNodeMatchesStartNode(node))
	{
		return true;
	}
	else
	{
		++steps;
		if (steps >= maxSteps)
			return false;

		GenerateAdjacentStates(node);

		// traverse every one of them
		for (int i = 0; i < 4; ++i)
		{
			if (node->neighbours[i] != nullptr && !(node->neighbours[i]->ifMarked))
			{
				if (DFS(node->neighbours[i], returnStack, steps, maxSteps))
				{
					return true;
				}
			}
		}

		returnStack->pop();
		return false;
	}
}

bool DFSIterative(Node* node, node_stack* returnStack, unsigned long &steps, unsigned long maxSteps)
{
	node_stack st; 
	Node* n;
	st.push(node);

	while (!st.empty())
	{
		++steps;
		if (steps >= maxSteps)
			return false;

		n = st.top();
		st.pop();

		if (n != nullptr && !n->ifMarked)
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

bool BFS(Node* node, node_stack* returnStack, unsigned long &steps, unsigned long maxSteps)
{
	queue<Node*> q;
	Node* n;

	q.push(node);
	node->ifMarked = true;

	while (!q.empty())
	{
		++steps;
		if (steps >= maxSteps)
			return false;

		n = q.front();
		q.pop();

		if (CheckIfNodeMatchesStartNode(n))
		{
			FillReturnStack(n, returnStack);
			return true;
		}

		GenerateAdjacentStates(n);
		for (int i = 0; i < 4; ++i)
		{
			if (n->neighbours[i] != nullptr && !(n->neighbours[i]->ifMarked))
			{
				q.push(n->neighbours[i]);
				n->neighbours[i]->ifMarked = true;
			}
		}
	}

	return false;
}

bool AStar(Node* node, node_stack* returnStack, unsigned long &steps, unsigned long maxSteps)
{
	node_priority_queue open;
	vector<Node*> closed;
	Node* tmpNode;
	Node* n;
	bool check;
	node->distance = 0;
	node->cost = GenerateCostManhattan(node) + node->distance;
	open.push(node);

	while (!open.empty())
	{
		++steps;
		if (steps >= maxSteps)
			return false;

		n = open.top();
		open.pop();

		if (CheckIfNodeMatchesStartNode(n))
		{
			FillReturnStack(n, returnStack);
			return true;
		}

		GenerateDumbAdjacentStates(n);
		for (int i = 0; i < 4; ++i)
		{
			if (n->neighbours[i] != nullptr)
			{
				check = CheckForExistingNode(n->neighbours[i], &closed);
				if (!check)
				{
					n->neighbours[i]->parent = n;
					n->neighbours[i]->distance = n->distance + 1;
					n->neighbours[i]->cost = GenerateCostManhattan(n->neighbours[i]) + n->neighbours[i]->distance;
					open.push(n->neighbours[i]);
					closed.push_back(n->neighbours[i]);
				}
				else
				{
					tmpNode = open.remove(n->neighbours[i]);
					if (tmpNode != nullptr)
					{
						n->neighbours[i] = tmpNode;
						n->neighbours[i]->parent = n;
						n->neighbours[i]->distance = n->distance + 1;
						n->neighbours[i]->cost = GenerateCostManhattan(n->neighbours[i]) + n->neighbours[i]->distance;

						if (tmpNode->cost > n->neighbours[i]->cost)
						{
							open.push(n->neighbours[i]);
						}
						else
						{
							open.push(tmpNode);
						}
					}
				}
			}
		}
	}

	return false;
}

inline void DeleteAllNodes()
{
	for (vector<Node*>::iterator it = globalNodeList.begin(); it != globalNodeList.end(); ++it)
	{
		delete (*it);
	}
	globalNodeList.clear();
}

inline void DeleteAllNodes(node_stack* st)
{
	for (vector<Node*>::iterator it = globalNodeList.begin(); it != globalNodeList.end(); ++it)
	{
		if (!(st->containsPtr(*it)))
			delete (*it);
	}
	globalNodeList.clear();
}

bool Search(unsigned char difficulty, unsigned char method, unsigned long maxSteps, node_stack* returnStack, unsigned long &steps, unsigned long &path, double &timer)
{
	bool result;
	Node* start = GenerateStartNode();
	ShuffleNode(start, difficulty);

	switch (method)
	{
	case 2:
		StartCounter();
		result = DFSIterative(start, returnStack, steps, maxSteps);
		timer = GetCounter();
		break;
	case 0:
		StartCounter();
		result = BFS(start, returnStack, steps, maxSteps);
		timer = GetCounter();
		break;
	case 1:
		StartCounter();
		result = AStar(start, returnStack, steps, maxSteps);
		timer = GetCounter();
		break;
	default:
		cout << "ERROR: Wrong method ID given to Search() function.";
		result = false;
		break;
	}

	DeleteAllNodes(returnStack);

	if (returnStack->size() == 0)
		path = 0;
	else
		path = returnStack->size() - 1;
	return result;
}

//////////////////////////////////////////////

inline void TraverseAndPrintPath(node_stack* returnStack)
{
	Node* ours;
	while (!returnStack->empty())
	{
		ours = returnStack->top();
		PrintNode(ours);
		returnStack->pop();
	}
}


//////////////////////////////////////////////

inline void GenerateTestNode(Node* node)
{
	node->board[0][0] = 1;
	node->board[0][1] = 2;
	node->board[0][2] = 3;
	node->board[0][3] = 4;
	node->board[1][0] = 5;
	node->board[1][1] = 6;
	node->board[1][2] = 7;
	node->board[1][3] = 8;
	node->board[2][0] = 9;
	node->board[2][1] = 14;
	node->board[2][2] = CURRENT_FIELD;
	node->board[2][3] = 10;
	node->board[3][0] = 13;
	node->board[3][1] = 11;
	node->board[3][2] = 15;
	node->board[3][3] = 12;
	node->currentX = 2;
	node->currentY = 2;
}

int main()
{
	unsigned int testsPerDifficultyLevel = 5;
	unsigned int maxDifficultyLevel = 10;
	unsigned char maxAlgorithmID = 3;
	unsigned long maxSteps = 100000;

	cout << "Maximum difficulty level : ";
	cin >> maxDifficultyLevel;

	cout << "Number of tests per difficulty level : ";
	cin >> testsPerDifficultyLevel;

	cout << "Maximum steps of algorithm : ";
	cin >> maxSteps;

	Tester tester(testsPerDifficultyLevel, maxDifficultyLevel, maxAlgorithmID, maxSteps, Search);

	tester.StartTests();

	string path = "results.txt";
	tester.WriteResults(&path);
	/*
	double timer;
	unsigned long steps = 0, path = 0;

	cout << "Searching starts... " << endl << endl;
	node_stack returnStack;
	Search(10, 3, &returnStack, steps, path, timer);

	TraverseAndPrintPath(&returnStack);
	cout << "Searching finished." << endl
		<< "Time taken: " << timer << " ms" << endl
		<< "Steps taken: " << steps << endl
		<< "Solution path length: " << path << endl 
		<< endl;
*/
	getch();

	return 0;
}