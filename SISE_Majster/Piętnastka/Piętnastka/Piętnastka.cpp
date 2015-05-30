// Piêtnastka.cpp : Defines the entry point for the console application.
// REPOSITORY VERSION
//

#include<iostream>
#include<iomanip>
#include<cstdlib>
#include<stack>
#include<queue>
#include<map>
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
	short i1 = 0;
	short i2 = 0;
	short i3 = 0;
	short i4 = 0;

public:
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
	int neighbours[4];
	int distances[16];
	int emptyI = 3;
	int emptyJ = 3;
	int direction = -1;
	int weight = 0;
	unsigned int id = 0;
	riddle riddleState;
	vector<mapState> nodes;
	vector<mapState> parent;

public:
	bool CheckSolved()
	{
		if ((riddleState.i1 == 4660) && (riddleState.i2 == 22136) &&
			(riddleState.i3 == -25924) && (riddleState.i4 == -8464))
			return true;
		else
			return false;
	}

	void operator==(const mapState& s)
	{
		neighbours[0] = s.neighbours[0];
		neighbours[1] = s.neighbours[1];
		neighbours[2] = s.neighbours[2];
		neighbours[3] = s.neighbours[3];
		emptyI = s.emptyI;
		emptyJ = s.emptyJ;
		direction = s.direction;
		id = s.id;
		riddleState = s.riddleState;
		nodes = s.nodes;
		parent = s.parent;
	}
};

mapState state;
bool isSolved = false;

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

void Move(int direction, int &emptyI, int &emptyJ)
{
	int tmp;
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
		//down
		if (emptyI < 3)
		{
			tmp = base[emptyI][emptyJ];
			base[emptyI][emptyJ] = base[emptyI + 1][emptyJ];
			base[emptyI + 1][emptyJ] = tmp;
			emptyI++;
		}
		break;
	case 2:
		//left
		if (emptyJ > 0)
		{
			tmp = base[emptyI][emptyJ];
			base[emptyI][emptyJ] = base[emptyI][emptyJ - 1];
			base[emptyI][emptyJ - 1] = tmp;
			emptyJ--;
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

void GetNeighbours(int &emptyI, int &emptyJ, int a[])
{
	if (emptyI > 0) //neighbour up
	{
		a[0] = base[emptyI - 1][emptyJ];
	}
	if (emptyI < 3) //neighbour down
	{
		a[1] = base[emptyI + 1][emptyJ];
	}
	if (emptyJ > 0) //neighbour left
	{
		a[2] = base[emptyI][emptyJ - 1];
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
		cout << oldDirection << ", " << direction << endl;
		oldDirection = direction;
	}
	return true;
}

//TODO: Generate Riddle with Pre-check algorithm [BFS i z grafu wyci¹gn¹æ losowy stan dla danego poziomu]

unsigned int GenerateID(map<unsigned int, mapState> &m)
{
	unsigned int id;
	map<unsigned int, mapState>::iterator it = m.begin();

	id = m.size() - 1;
	for (it = m.begin(); it != m.end(); it++)
	{
		if (it->first == id)
			id++;
	}

	return id;
}

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

void SetDistances(int table[])
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

void SetWeight(mapState &current, int table[])
{
	for (int i = 0; i < 16; ++i)
	{
		current.weight += table[i];
	}
}

void CheckChanges(mapState &current, mapState &newState)
{
	for (int i = 0; i < 15; ++i)
	{
		if (current.distances[i] != newState.distances[i])
		{
			if (current.distances[i] > newState.distances[i])
				newState.weight = newState.distances[i] - 1;
			else
				newState.weight = newState.distances[i] + 1;
		}
	}
}

bool operator>(const mapState& t, const mapState& s)
{
	if (t.weight > s.weight)
		return true;
	else
		return false;
}

bool operator<(const mapState&t, const mapState& s)
{
	if (t.weight < s.weight)
		return true;
	else
		return false;
}

//ALGORITHMS//

bool DFS(mapState &root) //TO DO: clean and try to achieve the shortest path by walking through ancestors
{
	unsigned int steps = 0;
	unsigned int id = 0;
	stack<mapState> stateStack;
	map<unsigned int, mapState> visited;
	stateStack.push(root);
	visited.insert(pair<unsigned int, mapState>(root.id, root));
	while (!stateStack.empty())
	{
		mapState currentState = stateStack.top();
		stateStack.pop();
		DecodeRiddle(currentState.riddleState);
		steps++;
		//processing node
		if (currentState.CheckSolved())
		{
			std::cout << "Riddle solved using " << steps << " steps." << endl;
			isSolved = true;
			return true;
		}
		else
		{
			//generate neighbours
			for (int i = 0; i < 4; ++i)
			{
				if (currentState.neighbours[i] == 0)
					continue;
				else
				{
					mapState newState;
					DecodeRiddle(currentState.riddleState);
					newState.neighbours[0] = 0;
					newState.neighbours[1] = 0;
					newState.neighbours[2] = 0;
					newState.neighbours[3] = 0;
					newState.emptyI = visited.at(currentState.id).emptyI;
					newState.emptyJ = visited.at(currentState.id).emptyJ;
					Move(i, newState.emptyI, newState.emptyJ);
					newState.riddleState = CodeRiddle();
					GetNeighbours(newState.emptyI, newState.emptyJ, newState.neighbours);
					newState.direction = i;
					id++;
					for (unsigned int k = 0; k < visited.size(); ++k)
					{
						if (newState.riddleState.operator==(visited.at(k).riddleState))
						{
							newState.id = visited.at(k).id;
							id--;
							break;
						}
						else
						{
							newState.id = id;
						}
					}
					currentState.nodes.push_back(newState);
				}
			}
			//for each neighbour in currentState make the algorithm
			for (int j = 0; j < currentState.nodes.capacity(); ++j)
			{
				if (visited.find(currentState.nodes[j].id) != visited.end())
					continue;
				else
				{
					stateStack.push(currentState.nodes[j]);
					visited.insert(pair<unsigned int, mapState>(currentState.nodes[j].id, currentState.nodes[j]));
				}
			}
		}
	}
	return false;
}

bool BFS(mapState &root) //TODO: optimalize (throws bad_alloc when difficulty of riddle is high :<)
{
	unsigned int steps = 0;
	unsigned int path = 0;
	unsigned int id = 0;
	queue<mapState> stateQueue;
	map<unsigned int, mapState> visited;
	stateQueue.push(root);
	visited.insert(pair<unsigned int, mapState>(root.id, root));
	while (!stateQueue.empty())
	{
		mapState currentState = stateQueue.front();
		stateQueue.pop();
		DecodeRiddle(currentState.riddleState);
		//ShowMatrix();
		steps++;
		//processing node
		if (currentState.CheckSolved())
		{
			if (currentState.parent.size() != 0)
			{
				mapState tmp;
				do
				{
					tmp = currentState.parent[0];
					currentState.operator==(tmp);
					path++;
					steps++;
				} while (!currentState.riddleState.operator==(root.riddleState));
			}
			std::cout << "Riddle solved using " << steps << " steps." << endl;
			std::cout << "Shortest path counts " << path << " steps." << endl;
			isSolved = true;
			return true;
		}
		else
		{
			//generate neighbours
			for (int i = 0; i < 4; ++i)
			{
				if (currentState.neighbours[i] == 0)
					continue;
				else
				{
					mapState newState;
					mapState oldState;
					oldState.operator==(currentState);
					DecodeRiddle(currentState.riddleState);
					newState.neighbours[0] = 0;
					newState.neighbours[1] = 0;
					newState.neighbours[2] = 0;
					newState.neighbours[3] = 0;
					newState.emptyI = visited.at(currentState.id).emptyI;
					newState.emptyJ = visited.at(currentState.id).emptyJ;
					newState.parent.push_back(oldState);
					Move(i, newState.emptyI, newState.emptyJ);
					newState.riddleState = CodeRiddle();
					GetNeighbours(newState.emptyI, newState.emptyJ, newState.neighbours);
					newState.direction = i;
					id++;
					for (unsigned int k = 0; k < visited.size(); ++k)
					{
						if (newState.riddleState.operator==(visited.at(k).riddleState))
						{
							newState.id = visited.at(k).id;
							id--;
							break;
						}
						else
						{
							newState.id = id;
						}
					}
					//cout << newState.id << " - his parent: " << newState.previousState->id << endl;
					currentState.nodes.push_back(newState);
				}
			}
			//for each neighbour in currentState make the algorithm
			for (int j = 0; j < currentState.nodes.size(); ++j)
			{
				if (visited.find(currentState.nodes[j].id) != visited.end())
					continue;
				else
				{
					stateQueue.push(currentState.nodes[j]);
					visited.insert(pair<unsigned int, mapState>(currentState.nodes[j].id, currentState.nodes[j]));
				}
			}
		}
	}
	return false;
}

bool ASTAR(mapState &root)
{
	unsigned int steps = 0;
	unsigned int id = 0;
	priority_queue<mapState, std::vector<mapState>, std::greater<mapState>> stateQueue;
	map<unsigned int, mapState> visited;
	stateQueue.push(root);
	visited.insert(pair<unsigned int, mapState>(root.id, root));
	while (!stateQueue.empty())
	{
		mapState currentState = stateQueue.top();
		stateQueue.pop();
		DecodeRiddle(currentState.riddleState);
		//ShowMatrix();
		steps++;
		//system("CLS");
		//cout << steps;
		//processing node
		if (currentState.CheckSolved())
		{
			int path = 0;
			if (currentState.parent.size() != 0)
			{
				mapState tmp;
				do
				{
					tmp = currentState.parent[0];
					currentState.operator==(tmp);
					path++;
					steps++;
				} while (!currentState.riddleState.operator==(root.riddleState));
			}
			std::cout << "Riddle solved using " << steps << " steps." << endl;
			std::cout << "Shortest path counts " << path << " nodes." << endl;
			isSolved = true;
			return true;
		}
		else
		{
			//generate neighbours
			for (int i = 0; i < 4; ++i)
			{
				if (currentState.neighbours[i] == 0)
					continue;
				else
				{
					mapState newState;
					mapState oldState;
					oldState.operator==(currentState);
					DecodeRiddle(currentState.riddleState);
					newState.neighbours[0] = 0;
					newState.neighbours[1] = 0;
					newState.neighbours[2] = 0;
					newState.neighbours[3] = 0;
					newState.emptyI = visited.at(currentState.id).emptyI;
					newState.emptyJ = visited.at(currentState.id).emptyJ;
					newState.parent.push_back(oldState);
					Move(i, newState.emptyI, newState.emptyJ);
					newState.riddleState = CodeRiddle();
					GetNeighbours(newState.emptyI, newState.emptyJ, newState.neighbours);
					newState.direction = i;
					SetDistances(newState.distances);
					SetWeight(newState, newState.distances);
					//CheckChanges(currentState, newState);
					newState.weight += currentState.weight + 1;
					id++;
					for (unsigned int k = 0; k < visited.size(); ++k)
					{
						if (newState.riddleState.operator==(visited.at(k).riddleState))
						{
							newState.id = visited.at(k).id;
							if (newState.weight < visited.at(k).weight)
							{
								visited.at(k).weight = newState.weight;
								stateQueue.push(visited.at(k));
							}
							id--;
							break;
						}
						else
						{
							newState.id = id;
						}					
					}
					//cout << newState.id << " - his parent: " << newState.previousState->id << endl;
					currentState.nodes.push_back(newState);
				}
			}
			//for each neighbour in currentState make the algorithm
			for (int j = 0; j < currentState.nodes.size(); ++j)
			{
				if (visited.find(currentState.nodes[j].id) != visited.end())
					continue;
				else
				{
					stateQueue.push(currentState.nodes[j]);
					visited.insert(pair<unsigned int, mapState>(currentState.nodes[j].id, currentState.nodes[j]));
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
	if (!GenerateRiddle(20))
		return 0;

	int direction;
	int steps = 0;
	ShowMatrix();
	cout << endl;
	GetNeighbours(state.emptyI, state.emptyJ, state.neighbours);
	mapState state0;
	state.riddleState = CodeRiddle();
	SetDistances(state.distances);
	state0.operator==(state);

	if (!ASTAR(state))
	{
		std::cout << "Something went wrong" << endl;
	}
	//ShowMatrix();

	cout << endl;

	state.operator==(state0);
	DecodeRiddle(state0.riddleState);
	ShowMatrix();

	if (!BFS(state))
	{
		std::cout << "Something went wrong" << endl;
	}

	cout << endl;

	system("PAUSE");
	return 0;
}

