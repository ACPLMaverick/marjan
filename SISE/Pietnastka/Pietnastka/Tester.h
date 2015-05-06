#pragma once
#include "Header.h"
#include <fstream>
#include <map>

class node_stack;

typedef bool(*SearchFunc)(unsigned char, unsigned char, unsigned long, node_stack*, unsigned long&, unsigned long&, double&);

struct TesterModule
{
	unsigned long stepsTaken = 0;
	unsigned long pathLength = 0;
	double timeTaken = 0.0;
	node_stack* stepStack;
	unsigned char algorithmID;
	unsigned char difficultyLevel;
	bool ifSucceeded;

	~TesterModule()
	{
		if (stepStack != nullptr)
		{
			Node* tempNode;
			while (!stepStack->empty())
			{
				tempNode = stepStack->top();
				stepStack->pop();
				if (tempNode != nullptr)
					delete tempNode;
			}
			delete stepStack;
		}
	}
};

class Tester
{
private:
	unsigned long maxSteps;

	TesterModule**** testerBlock;
	SearchFunc search;

	std::map<unsigned char, std::string> nameMap;

	unsigned int testsPerDifficultyLevel;
	unsigned int maxDifficultyLevel;
	unsigned char maxAlgorithmID;

public:
	Tester(unsigned int tpd, unsigned int mdl, unsigned char maid, unsigned long maxSt, SearchFunc sfunc);
	~Tester();

	void StartTests();
	void WriteResults();
	void WriteResults(std::string* filename);
};

