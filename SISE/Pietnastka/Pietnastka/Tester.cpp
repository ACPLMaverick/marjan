#include "Tester.h"


Tester::Tester(unsigned int tpd, unsigned int mdl, unsigned char maid, unsigned long maxSt, SearchFunc sfunc)
{
	this->testsPerDifficultyLevel = tpd;
	this->maxDifficultyLevel = mdl;
	this->maxAlgorithmID = maid;
	this->search = sfunc;
	this->maxSteps = maxSt;

	(this->testerBlock) = new TesterModule***[maxDifficultyLevel];
	for (int i = 0; i < maxDifficultyLevel; ++i)
	{
		this->testerBlock[i] = new TesterModule**[testsPerDifficultyLevel];
		for (int j = 0; j < testsPerDifficultyLevel; ++j)
		{
			this->testerBlock[i][j] = new TesterModule*[maxAlgorithmID];
			for (int k = 0; k < maxAlgorithmID; ++k)
			{
				(this->testerBlock[i][j][k]) = new TesterModule;
			}
		}
	}

	this->nameMap[2] = "DFS";
	this->nameMap[0] = "BFS";
	this->nameMap[1] = " A\*";
	this->nameMap[3] = "Unknown";
}


Tester::~Tester()
{
	for (int i = 0; i < maxDifficultyLevel; ++i)
	{
		for (int j = 0; j < testsPerDifficultyLevel; ++j)
		{
			for (int k = 0; k < maxAlgorithmID; ++k)
			{
				delete (this->testerBlock[i][j][k]);
			}
			delete[] this->testerBlock[i][j];
		}
		delete[] this->testerBlock[i];
	}
	delete[] this->testerBlock;

	this->nameMap.erase(nameMap.begin(), nameMap.end());
}

void Tester::StartTests()
{
	std::cout << "TESTER: Tests started." << std::endl
<< "=============================" << std::endl << std::endl;
	bool result;

	for (int i = 0; i < maxDifficultyLevel; ++i)
	{
		for (int j = 0; j < testsPerDifficultyLevel; ++j)
		{
			for (int k = 0; k < maxAlgorithmID; ++k)
			{
				
				std::cout << "TESTER: Testing " << (nameMap[k > 3 ? 3 : k]).c_str()
					<< ", difficulty " << i + 1 << ", test no. " << j << " ....... ";

				testerBlock[i][j][k]->stepStack = new node_stack;
				testerBlock[i][j][k]->difficultyLevel = i;
				testerBlock[i][j][k]->algorithmID = k;

				result = search(i + 1, k, maxSteps,
					testerBlock[i][j][k]->stepStack,
					testerBlock[i][j][k]->stepsTaken,
					testerBlock[i][j][k]->pathLength,
					testerBlock[i][j][k]->timeTaken);

				testerBlock[i][j][k]->ifSucceeded = result;

				if (result)
				{
					std::cout << "Finished." << std::endl;
				}
				else
				{
					std::cout << "Failed." << std::endl;
				}
				
			}
		}
	}
}

void Tester::WriteResults()
{
	for (int i = 0; i < maxDifficultyLevel; ++i)
	{
		for (int j = 0; j < testsPerDifficultyLevel; ++j)
		{
			for (int k = 0; k < maxAlgorithmID; ++k)
			{
				std::cout << "TESTER: Test " << (nameMap[k > 3 ? 3 : k]).c_str()
					<< ", difficulty " << i + 1 << ", test no. " << j << std::endl
					<< "	Time taken: " << testerBlock[i][j][k]->timeTaken << " ms, " 
					<< "Steps taken: " << testerBlock[i][j][k]->stepsTaken << ", "
					<< "Path length: " << testerBlock[i][j][k]->pathLength << ", ";

				if (testerBlock[i][j][k]->ifSucceeded)
				{
					std::cout << "SUCCEEDED." << std::endl;
				}
				else
				{
					std::cout << "FAILED." << std::endl;
				}
			}
		}
	}
}

void Tester::WriteResults(std::string* filename)
{
	std::ofstream strm(*filename, std::ofstream::out);
	for (int i = 0; i < maxAlgorithmID; ++i)
	{
		strm << "Algorithm:  " << nameMap[i > 3 ? 3 : i].c_str() << std::endl << std::endl;
		for (int j = 0; j < maxDifficultyLevel; ++j)
		{
			strm << "Difficulty: " << j + 1 << std::endl;
			for (int k = 0; k < testsPerDifficultyLevel; ++k)
			{
				strm << testerBlock[j][k][i]->timeTaken << ", "
					<< testerBlock[j][k][i]->stepsTaken << ", "
					<< testerBlock[j][k][i]->pathLength << ", ";

				if (testerBlock[j][j][i]->ifSucceeded)
				{
					strm << "SUCCEEDED." << std::endl;
				}
				else
				{
					strm << "FAILED." << std::endl;
				}
			}
			strm << std::endl;
		}
		strm << "=========================" << std::endl;
	}

	strm.close();
}