#include <conio.h>
#include <cstdlib>
#include <iostream>
#include <time.h>
#include <vector>
#include <stack>

#include "Header.h"

using namespace std;

///////////////////////////////////////////////

struct Node
{
	unsigned char currentX;
	unsigned char currentY;
	unsigned char board[RIDDLE_SIZE][RIDDLE_SIZE];

	Node* neighbours[4];
	bool ifMarked = false;

	bool operator==(const Node &other)
	{
		if (currentX != other.currentX || currentY != other.currentY)
			return false;

		int ctr = 1;
		for (int i = 0; i < RIDDLE_SIZE; ++i)
		{
			for (int j = 0; j < RIDDLE_SIZE; ++j)
			{
				if (board[i][j] != other.board[i][j])
					return false;
			}
		}
		return true;
	}

	bool operator!=(const Node &other)
	{
		return !(operator==(other));
	}

	Node& operator=(const Node &first)
	{
		currentX = first.currentX;
		currentY = first.currentY;

		for (int i = 0; i < RIDDLE_SIZE; ++i)
		{
			for (int j = 0; j < RIDDLE_SIZE; ++j)
			{
				board[i][j] = first.board[i][j];
			}
		}

		for (int i = 0; i < 4; ++i)
		{
			neighbours[i] = first.neighbours[i];
		}

		ifMarked = first.ifMarked;

		return *this;
	}
};

//////////////////////////////////////////////
////// GLOBALS

vector<Node*> globalNodeList;

//////////////////////////////////////////////

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

bool CreateNodeFromMove(const Node* oldState, Node* newState, unsigned int dir)
{
	(*newState) = (*oldState);
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

	return true;
}

Node* CreateNode()
{
	Node* state = new Node;

	globalNodeList.push_back(state);

	return state;
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
		// create adjacent nodes
		Node* parent = returnStack->top();

		for (int i = 0; i < 4; ++i)
		{
			Node* dir = new Node;
			CreateNodeFromMove(node, dir, i);

			// compare with parent
			if (dir == parent)
			{
				node->neighbours[i] = parent;
				delete dir;
			}
			else
			{
				bool assigned = false;
				for (vector<Node*>::iterator it = globalNodeList.begin(); it != globalNodeList.end(); ++it)
				{
					if (dir == (*it))
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

void Search(Node* start, unsigned int method, stack<Node*>* returnStack)
{
	switch (method)
	{
	case 0:
		DFS(start, returnStack);
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

	ShuffleNode(startNode, 1);

	cout << "Shuffled start state: " << endl << endl;
	PrintNode(startNode);

	stack<Node*> returnStack;
	Search(startNode, 0, &returnStack);

	cout << "Solution path: " << endl << endl;
	TraverseAndPrintPath(&returnStack);

	getch();

	DeleteAllNodes();

	return 0;
}