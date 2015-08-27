// Piêtnastka.cpp : Defines the entry point for the console application.
// REPOSITORY VERSION
//

#include<iostream>
#include<iomanip>
#include<cstdlib>
#include<stack>
#include<queue>
#include<list>
#include<string>
#include<vector>
#include<functional>

using namespace std;

//GLOBALS//

short base[4][4] = {
	{ 1, 2, 3, 4 },
	{ 5, 6, 7, 8 },
	{ 9, 10, 11, 12 },
	{ 13, 14, 15, 0 }
};

struct riddle
{
	short i1;
	short i2;
	short i3;
	short i4;

public:
	riddle()
	{
		i1 = 0;
		i2 = 0;
		i3 = 0;
		i4 = 0;
	}

	riddle(const riddle& r)
	{
		i1 = r.i1;
		i2 = r.i2;
		i3 = r.i3;
		i4 = r.i4;
	}

	bool operator==(const riddle &r)
	{
		if ((i1 == r.i1) && (i2 == r.i2) && (i3 == r.i3) && (i4 == r.i4))
			return true;
		else
			return false;
	}
};

struct mapState
{
	short neighbours[4];
	short distances[16];
	short emptyI;
	short emptyJ;
	short weight;
	short depth;
	string id;
	riddle riddleState;
	mapState* parent;

public:
	mapState()
	{
		for (int i = 0; i < 4; ++i)
		{
			neighbours[i] = 0;
		}
		for (int j = 0; j < 16; ++j)
		{
			distances[j] = 0;
		}
		emptyI = 3;
		emptyJ = 3;
		weight = 0;
		depth = 0;
		id = "Key";
		parent = nullptr;
	}

	mapState(const mapState& m)
	{
		for (int i = 0; i < 4; ++i)
		{
			neighbours[i] = m.neighbours[i];
		}
		for (int j = 0; j < 16; ++j)
		{
			distances[j] = m.distances[j];
		}
		emptyI = m.emptyI;
		emptyJ = m.emptyJ;
		weight = m.weight;
		depth = m.depth;
		id = m.id;
		riddleState = m.riddleState;
		parent = m.parent;
	}

	bool CheckSolved()
	{
		if ((riddleState.i1 == 4660) && (riddleState.i2 == 22136) &&
			(riddleState.i3 == -25924) && (riddleState.i4 == -8464))
			return true;
		else
			return false;
	}

	void GenerateKey()
	{
		for (int i = 0; i < 4; ++i)
		{
			for (int j = 0; j < 4; ++j)
			{
				id += to_string(base[i][j]);
			}
		}
	}
};

mapState state;
bool isSolved = false;
vector<mapState*> pointers;

//METHODS//

void ShowMatrix()
{
	for (int i = 0; i < 4; ++i)
	{
		for (int j = 0; j < 4; ++j)
		{
			if (base[i][j] == 0)
			{
				std::cout << setw(5) << " ";
			}
			else
			{
				std::cout << setw(4) << base[i][j] << " ";
			}
		}
		std::cout << endl;
	}
}

void Move(short direction, short &emptyI, short &emptyJ)
{
	short tmp;
	switch (direction)
	{
	case 0:
		//up
		if (emptyI > 0)
		{
			tmp = base[emptyI][emptyJ];
			base[emptyI][emptyJ] = base[emptyI - 1][emptyJ];
			base[emptyI - 1][emptyJ] = tmp;
			emptyI--;
		}
		break;
	case 1:
		//left
		if (emptyJ > 0)
		{
			tmp = base[emptyI][emptyJ];
			base[emptyI][emptyJ] = base[emptyI][emptyJ - 1];
			base[emptyI][emptyJ - 1] = tmp;
			emptyJ--;
		}
		break;
	case 2:
		//down
		if (emptyI < 3)
		{
			tmp = base[emptyI][emptyJ];
			base[emptyI][emptyJ] = base[emptyI + 1][emptyJ];
			base[emptyI + 1][emptyJ] = tmp;
			emptyI++;
		}
		break;
	case 3:
		//right
		if (emptyJ < 3)
		{
			tmp = base[emptyI][emptyJ];
			base[emptyI][emptyJ] = base[emptyI][emptyJ + 1];
			base[emptyI][emptyJ + 1] = tmp;
			emptyJ++;
		}
		break;
	}
}

void GetNeighbours(short &emptyI, short &emptyJ, short* a)
{
	if (emptyI > 0) //neighbour up
	{
		a[0] = base[emptyI - 1][emptyJ];
	}
	if (emptyJ > 0) //neighbour left
	{
		a[1] = base[emptyI][emptyJ - 1];
	}
	if (emptyI < 3) //neighbour down
	{
		a[2] = base[emptyI + 1][emptyJ];
	}
	if (emptyJ < 3) //neighbour right
	{
		a[3] = base[emptyI][emptyJ + 1];
	}
}

void ShowNeighbours(mapState &state)
{
	for (int i = 0; i < 4; i++)
	{
		//if (state.neighbours[i] == 0)
		//	continue;
		//else
			std::cout << state.neighbours[i] << " ";
	}
	std::cout << endl;
}

bool GenerateRiddle(int steps)
{
	//generating riddle using set number of steps
	//move the empty (16) node number of times (steps) through the matrix
	int direction;
	int oldDirection = -1;
	for (int i = 0; i < steps; ++i)
	{
		do
		{
			direction = rand() % 4;
		} while (direction == (oldDirection + 2) % 4);
		Move(direction, state.emptyI, state.emptyJ);
		oldDirection = direction;
	}
	return true;
}

//unsigned int GenerateID(map<unsigned int, mapState> &m)
//{
//	unsigned int id;
//	map<unsigned int, mapState>::iterator it = m.begin();
//
//	id = m.size() - 1;
//	for (it = m.begin(); it != m.end(); it++)
//	{
//		if (it->first == id)
//			id++;
//	}
//
//	return id;
//}

riddle CodeRiddle()
{
	riddle u;
	for (int j = 0; j < 4; ++j)
	{
		u.i1 |= (base[0][j] << (12 - (4 * j)));
		u.i2 |= (base[1][j] << (12 - (4 * j)));
		u.i3 |= (base[2][j] << (12 - (4 * j)));
		u.i4 |= (base[3][j] << (12 - (4 * j)));
	}
	return u;
}

short ReadFrom(short data, short index, short size)
{
	short output;
	output = data & (((1 << size) - 1) << index);
 	return output >> index;
}

void DecodeRiddle(riddle &r)
{
	for (int i = 0; i < 4; ++i)
	{
		base[0][i] = ReadFrom(r.i1, 12 - (4 * i), 4); //read bits
		base[0][i] &= 0x000f;
		base[1][i] = ReadFrom(r.i2, 12 - (4 * i), 4);
		base[1][i] &= 0x000f;
		base[2][i] = ReadFrom(r.i3, 12 - (4 * i), 4);
		base[2][i] &= 0x000f;
		base[3][i] = ReadFrom(r.i4, 12 - (4 * i), 4);
		base[3][i] &= 0x000f;
	}
}

//METHODS NEW//

void SetDistances(short* table)
{
	int targetI, targetJ;
	for (int i = 0; i < 4; ++i)
	{
		for (int j = 0; j < 4; ++j)
		{
			if (base[i][j] == 0)
			{
				targetI = 3;
				targetJ = 3;
				table[15] = (abs(i - targetI)) + (abs(j - targetJ));
			}
			else
			{
				targetJ = (base[i][j] - 1) % 4;
				targetI = (base[i][j] - j) / 4;
				int it = base[i][j];
				table[it-1] = (abs(i - targetI)) + (abs(j - targetJ));
			}
		}
	}
}

void SetWeight(mapState &current, short* table)
{
	for (int i = 0; i < 16; ++i)
	{
		current.weight += table[i];
	}
}

struct mapStateCompare
{
	bool operator()(const mapState* t, const mapState* s) const
	{
		return t->weight > s->weight;
	}
};

bool Contains(list<string> *graph, string s)
{
	for (list<string>::iterator it = graph->begin(); it != graph->end(); ++it)
	{
		if ((*it) == s) return true;
	}
	return false;
}

//ALGORITHMS//

bool DFS(mapState &root)
{
	unsigned int steps = 0;
	unsigned int path = 0;
	stack<mapState*> stateStack;
	mapState* currentState;
	list<string> visited;
	stateStack.push(&root);
	visited.push_back(root.id);
	while (!stateStack.empty() || steps < 100000)
	{
		currentState = stateStack.top();
		stateStack.pop();
		DecodeRiddle(currentState->riddleState);
		steps++;
		//processing node
		if (currentState->CheckSolved())
		{
			clock_t startTime = std::clock();
			if (currentState->parent != nullptr)
			{
				mapState* tmp;
				do
				{
					tmp = currentState->parent;
					currentState = tmp;
					path++;
					//steps++;
				} while (!currentState->riddleState.operator==(root.riddleState));
			}
			std::cout << "DFS - Riddle solved using " << steps << " steps." << endl;
			std::cout << "Shortest path counts " << path << " steps." << endl;
			isSolved = true;
			visited.clear();
			while (!stateStack.empty())
			{
				stateStack.pop();
			}
			cout << ((std::clock() - startTime) / (double)CLOCKS_PER_SEC) * 1000.0f << "ms\n";
			return true;
		}
		else
		{
			//generate neighbours
			for (int i = 0; i < 4; ++i)
			{
				if (currentState->neighbours[i] == 0)
					continue;
				else
				{
					mapState* newState = new mapState();
					mapState* oldState = new mapState();

					pointers.push_back(newState);
					pointers.push_back(oldState);

					oldState = currentState;
					DecodeRiddle(currentState->riddleState);
					newState->emptyI = currentState->emptyI;
					newState->emptyJ = currentState->emptyJ;
					newState->parent = oldState;
					Move(i, newState->emptyI, newState->emptyJ);
					newState->riddleState = CodeRiddle();
					GetNeighbours(newState->emptyI, newState->emptyJ, newState->neighbours);
					newState->GenerateKey();
					newState->depth += oldState->depth;
					newState->depth++;

					if (!Contains(&visited, newState->id))
					{
						stateStack.push(newState);
						visited.push_back(newState->id);
					}
				}
			}
		}
	}
	return false;
}

bool BFS(mapState &root)
{
	unsigned int steps = 0;
	unsigned int path = 0;
	queue<mapState*> stateQueue;
	mapState* currentState;
	list<string> visited;
	stateQueue.push(&root);
	visited.push_back(root.id);
	while (!stateQueue.empty() || steps < 100000)
	{
		currentState = stateQueue.front();
		stateQueue.pop();
		DecodeRiddle(currentState->riddleState);
		//ShowMatrix();
		steps++;
		//processing node
		if (currentState->CheckSolved())
		{
			if (currentState->parent != nullptr)
			{
				mapState* tmp;
				do
				{
					tmp = currentState->parent;
					currentState = tmp;
					path++;
					//steps++;
				} while (!currentState->riddleState.operator==(root.riddleState));
			}
			std::cout << "BFS - Riddle solved using " << steps << " steps." << endl;
			std::cout << "Shortest path counts " << path << " steps." << endl;
			isSolved = true;
			visited.clear();
			while (!stateQueue.empty())
			{
				stateQueue.pop();
			}
			return true;
		}
		else
		{
			//generate neighbours
			for (int i = 0; i < 4; ++i)
			{
				if (currentState->neighbours[i] == 0)
					continue;
				else
				{
					mapState* newState = new mapState();
					mapState* oldState = new mapState();

					pointers.push_back(newState);
					pointers.push_back(oldState);

					oldState = currentState;
					DecodeRiddle(currentState->riddleState);
					newState->emptyI = currentState->emptyI;
					newState->emptyJ = currentState->emptyJ;
					newState->parent = oldState;
					Move(i, newState->emptyI, newState->emptyJ);
					newState->riddleState = CodeRiddle();
					GetNeighbours(newState->emptyI, newState->emptyJ, newState->neighbours);
					newState->GenerateKey();
					newState->depth += oldState->depth;
					newState->depth++;

					if (!Contains(&visited, newState->id))
					{
						stateQueue.push(newState);
						visited.push_back(newState->id);
					}
				}
			}
		}
	}
	return false;
}

bool ASTAR(mapState &root)
{
	unsigned int steps = 0;
	priority_queue<mapState*, std::vector<mapState*>, mapStateCompare> stateQueue;
	mapState* currentState;
	vector<mapState*> pointersLocal;
	list<string> visited;
	stateQueue.push(&root);
	visited.push_back(root.id);
	while (!stateQueue.empty())
	{
		currentState = stateQueue.top();
		stateQueue.pop();
		DecodeRiddle(currentState->riddleState);
		//ShowMatrix();
		steps++;
		//system("CLS");
		//cout << steps;
		//processing node
		if (currentState->CheckSolved())
		{
			int path = 0;
			if (currentState->parent != nullptr)
			{
				mapState* tmp;
				do
				{
					tmp = currentState->parent;
					currentState = tmp;
					path++;
					//steps++;
				} while (!currentState->riddleState.operator==(root.riddleState));
			}
			std::cout << "A* - Riddle solved using " << steps << " steps." << endl;
			std::cout << "Shortest path counts " << path << " steps." << endl;
			isSolved = true;
			while (!visited.empty())
			{
				visited.pop_back();
			}
			while (!stateQueue.empty())
			{
				stateQueue.pop();
			}
			for (int i = 0; i < pointersLocal.size(); ++i)
			{
				if (pointersLocal.at(i) != nullptr)
					delete pointersLocal.at(i);
			}

			return true;
		}
		else
		{
			//generate neighbours
			for (int i = 0; i < 4; ++i)
			{
				if (currentState->neighbours[i] == 0)
					continue;
				else
				{
					mapState* newState = new mapState();
					mapState* oldState = new mapState();

					pointersLocal.push_back(newState);
					pointersLocal.push_back(oldState);

					oldState = currentState;
					DecodeRiddle(currentState->riddleState);
					newState->emptyI = currentState->emptyI;
					newState->emptyJ = currentState->emptyJ;
					newState->parent = oldState;
					Move(i, newState->emptyI, newState->emptyJ);
					newState->riddleState = CodeRiddle();
					GetNeighbours(newState->emptyI, newState->emptyJ, newState->neighbours);
					newState->GenerateKey();
					newState->depth += oldState->depth;
					newState->depth++;

					SetDistances(newState->distances);
					SetWeight(*newState, newState->distances);
					newState->weight += oldState->weight + 1;

					if (!Contains(&visited, newState->id))
					{
						stateQueue.push(newState);
						visited.push_back(newState->id);
					}
				}
			}
		}
	}
	return false;
}

//MAIN//

int main(int argc, char* argv[])
{
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
	srand(time(NULL));
	int difficulty = 0;
	cout << "Set maximum depth: " << endl;
	cin >> difficulty;

	if (!GenerateRiddle(difficulty))
		return 0;

	int direction;
	int steps = 0;
	ShowMatrix();
	cout << endl;
	GetNeighbours(state.emptyI, state.emptyJ, state.neighbours);
	mapState state0, state1;
	state.riddleState = CodeRiddle();
	SetDistances(state.distances);
	state0 = state;

	clock_t startTime = std::clock();
	cout << "A* processing..." << endl;
	if (!ASTAR(state))
	{
		std::cout << "Something went wrong" << endl;
	}
	cout << "Total time: " << ((std::clock() - startTime) / (double)CLOCKS_PER_SEC) * 1000.0f << "ms\n";

	//ShowMatrix();

	cout << endl;

	state = state0;
	DecodeRiddle(state0.riddleState);

	startTime = std::clock();
	cout << "BFS processing..." << endl;
	if (!BFS(state))
	{
		std::cout << "Something went wrong" << endl;
	}
	cout << "Total time: " << ((std::clock() - startTime) / (double)CLOCKS_PER_SEC) * 1000.0f << "ms\n\n";

	state = state0;
	DecodeRiddle(state0.riddleState);

	startTime = std::clock();
	cout << "DFS processing..." << endl;
	if (!DFS(state))
	{
		std::cout << "Something went wrong" << endl;
	}
	cout << "Total time: " << ((std::clock() - startTime) / (double)CLOCKS_PER_SEC) * 1000.0f << "ms";

	for (int i = 0; i < pointers.size(); ++i)
	{
		if (pointers.at(i) != nullptr)
			delete pointers.at(i);
	}

	cout << endl;

	system("PAUSE");
	return 0;
}

