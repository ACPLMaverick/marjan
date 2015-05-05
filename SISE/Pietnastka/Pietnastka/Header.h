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
	unsigned char currentX;
	unsigned char currentY;
	unsigned char board[RIDDLE_SIZE][RIDDLE_SIZE];

	Node* parent;
	Node* neighbours[4];
	bool ifMarked = false;
	cost_t cost = UINT_MAX;
	cost_t distance = 0;

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
