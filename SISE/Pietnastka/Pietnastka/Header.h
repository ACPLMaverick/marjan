#pragma once

#include <conio.h>
#include <cstdlib>
#include <iostream>
#include <time.h>
#include <vector>
#include <stack>
#include <queue>

#define RIDDLE_SIZE 4
#define CURRENT_FIELD ' '
#define MOVE_UP 0
#define MOVE_DOWN 1
#define MOVE_LEFT 2
#define MOVE_RIGHT 3

typedef unsigned int cost_t;

struct Node
{
	Node* parent;
	Node* neighbours[4];
	cost_t cost = UINT_MAX;
	cost_t distance = 0;
	unsigned char board[RIDDLE_SIZE][RIDDLE_SIZE];
	unsigned char currentX;
	unsigned char currentY;
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

	bool operator<(const Node &other)
	{
		return cost < other.cost;
	}

	bool operator>(const Node &other)
	{
		return cost > other.cost;
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
		parent = first.parent;

		ifMarked = first.ifMarked;
		cost = first.cost;
		distance = first.distance;

		return *this;
	}
};

class NodeComparator
{
public:
	bool operator() (const Node* lv, const Node* rv) const
	{
		return lv->cost > rv->cost;
	}
};

class node_stack : public std::stack<Node*>
{
public:

	bool containsPtr(Node* node)
	{
		std::deque<Node*>::iterator end = c.end();
		for (std::deque<Node*>::iterator it = c.begin(); it != end; ++it)
		{
			if (node == (*it))
				return true;
		}

		return false;
	}
};

class node_priority_queue : public std::priority_queue<Node*, std::vector<Node*>, NodeComparator>
{
public:
	bool contains(Node* node)
	{
		for (std::vector<Node*>::iterator it = this->c.begin(); it != this->c.end(); ++it)
		{
			if ((*node) == (*(*it)))
				return true;
		}
		return false;
	}

	bool containsPtr(Node* node)
	{
		for (std::vector<Node*>::iterator it = this->c.begin(); it != this->c.end(); ++it)
		{
			if (node == (*it))
				return true;
		}
		return false;
	}

	bool containsLowerCost(Node* node)
	{
		for (std::vector<Node*>::iterator it = this->c.begin(); it != this->c.end(); ++it)
		{
			if (node->cost > (*it)->cost && node->distance == (*it)->distance)
				return true;
		}
		return false;
	}

	Node* remove(Node* node)
	{
		std::vector<Node*> tmpVec;
		Node* tmpNode = nullptr;
		bool found = false;

		while (!this->empty())
		{
			tmpNode = this->top();
			this->pop();

			if ((tmpNode) == (node))
			{
				found = true;
				break;
			}
			else
			{
				tmpVec.push_back(tmpNode);
			}
		}

		unsigned int vecSize = tmpVec.size();
		for (unsigned int i = 0; i < vecSize; ++i)
		{
			this->push(tmpVec[i]);
		}

		if (!found) tmpNode = nullptr;
		return tmpNode;
	}
};
