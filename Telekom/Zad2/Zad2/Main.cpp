#include <string>
#include <iostream>
#include <stdio.h>
#include <cstdlib>
#include <string.h>
#include <conio.h>
#include <fstream>

using namespace std;

string loadFile(string path);
void encryptFile();
void decryptFile();
void convertToBinary(unsigned int letter, unsigned int tab[]);

const int hTable[8][16] = { 
{ 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0 },
{ 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0 },
{ 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0 },
{ 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0 },
{ 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0 },
{ 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0 },
{ 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0 },
{ 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1 } 
};

const string PATH_LOAD = "picture2.bmp";
const string PATH_OUTPUT = "encrypted.txt";
const string PATH_DECRYPTED = "decrypted.bmp";

int main()
{
	encryptFile();
	_getch();
	decryptFile();

	_getch();
	return 0;
}

void encryptFile()
{
	cout << "KODOWANIE PLIKU...\n";
	string myFile = loadFile(PATH_LOAD);
	int size = myFile.length();
	unsigned int* chars = new unsigned int[size+1];
	unsigned int** bytes = new unsigned int*[size+1];
	for (int i = 0; i < size + 1; i++) bytes[i] = new unsigned int[16];
	unsigned int charBuffer[8];

	// konwersja stringu do tablicy charów
	//strcpy(chars, myFile.c_str());
	for (int i = 0; i < size; i++) chars[i] = myFile.at(i);

	// wpisywanie bajtów pliku do tablicy w formie binarnej
	for (int i = 0; i < size; i++)
	{
		convertToBinary(chars[i], charBuffer);
		for (int j = 0; j < 8; j++)
		{
			bytes[i][j] = charBuffer[j];
		}
		// bity kontroli
		for (int j = 8; j < 16; j++)
		{
			bytes[i][j] = 0;
			for (int k = 0; k < 8; k++)
			{
				bytes[i][j] += charBuffer[k] * hTable[j-8][k];
			}

			bytes[i][j] %= 2;
		}
	}

	/*
	// wypisywanie tablicy na konsole - niepotrzebne
	for (int i = 0; i < size + 1; i++)
	{
		for (int j = 0; j < 16; j++)
		{
			if (j == 8) cout << " ";
			cout << bytes[i][j];
		}
		cout << endl;
	}
	*/

	// zapisywanie do pliku
	ofstream outputFile;
	outputFile.open(PATH_OUTPUT);
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < 16; j++)
		{
			outputFile << bytes[i][j];
		}
		outputFile << endl;
	}

	delete[] chars;
	for (int i = 0; i < size; i++) delete[] bytes[i];
	delete[] bytes;
	
	cout << "PLIK " + PATH_LOAD + " ZAKODOWANY DO PLIKU " + PATH_OUTPUT + "\n";
}

void decryptFile()
{
	cout << "DEKODOWANIE PLIKU " + PATH_OUTPUT + "\n";

	string myFile = loadFile(PATH_OUTPUT);
	int size = myFile.length()/16;
	unsigned int wordArray[16];
	unsigned int errorArray[8];
	char currentChar = 0;
	bool isError = false;
	int errorCount = 0;
	unsigned int** bytes = new unsigned int*[size];
	for (int i = 0; i < size; i++) bytes[i] = new unsigned int[16];

	// wrzucanie s³ów bitowych do tablicy
	int myChar;
	for (int i = 0, j = 0, k = 0; i < myFile.length(); i++)
	{
		myChar = myFile.at(i);
		if (myChar != '\n')
		{
			bytes[j][k] = myChar - 48;
			k++;
		}
		else
		{
			j++;
			k = 0;
		}
	}

	// teraz mamy bity w tablicy trzeba siê wzi¹æ za odkodowywanie :D

	// otwarcie pliku do zapisu
	ofstream outputFile;
	outputFile.open(PATH_DECRYPTED);
	
	// mno¿enie z hTable i sprawdzanie czy = 0 - zgodnie z instrukcj¹, jeœli bêd¹ same 0 to nie ma b³êdów
	for (int i = 0; i < size-2; i++)
	{
		for (int j = 0; j < 16; j++)
		{
			wordArray[j] = bytes[i][j];
		}

		for (int j = 0; j < 8; j++) errorArray[j] = 0;

		for (int j = 0; j < 8; j++)
		{
			for (int k = 0; k < 16; k++)
			{
				errorArray[j] += (wordArray[k] * hTable[j][k]);
			}
			errorArray[j] %= 2;

			if (errorArray[j] == 1) errorCount = 1;
		}

		// sprawdzanie i korekcja b³êdów

		if (errorCount != 0)
		{
			//cout << "ERROR! ";
			isError = false;

			for (int j = 0; j < 15; j++)
			{
				for (int k = j + 1; k < 16; k++)
				{
					isError = true;
					for (int m = 0; m < 8; m++)
					{
						// znajdowanie miejsc z b³êdem i naprawa dla dwóch b³êdów
						if (errorArray[m] != hTable[m][j] ^ hTable[m][k])
						{
							isError = false;
							break;
						}
					}
					if (isError)
					{
						cout << "DWA bledy znalezione: w " << i+1 << " znaku, na " << j+1 << " i " << k+1 << " bicie.\n";
						if (wordArray[j] == 0) wordArray[j] = 1;
						else wordArray[j] = 0;
						if (wordArray[k] == 0) wordArray[k] = 1;
						else wordArray[k] = 0;
						errorCount = 2;
						j = 16;
						break;
					}
				}
			}

			if (errorCount == 1)
			{
				for (int j = 0; j < 16; j++)
				{
					for (int k = 0; k < 8; k++)
					{
						if (errorArray[k] != hTable[k][j]) break;
						// znajdowanie miejsca z b³êdem i naprawa b³êdu dla 1 b³êdu

						// to siê wykona kiedy nie bêd¹ siê ró¿niæ
						if (k == 7)
						{
							cout << "JEDEN blad znaleziony: w " << i + 1 << " znaku, na " << j + 1 << " bicie.\n";
							if (wordArray[j] == 0) wordArray[j] = 1;
							else wordArray[j] = 0;
							j = 16;
						}
					}
				}
				errorCount = 0;
			}
		}

		// konwersja i zapisywanie znaku do pliku
		currentChar = 0;
		for (int j = 0, pow = 128; j < 16; j++, pow/=2)
		{
			currentChar += pow * wordArray[j];
		}
		outputFile << currentChar;
	}

	outputFile.close();
	cout << "PLIK " + PATH_OUTPUT + " ODKODOWANY DO PLIKU " + PATH_DECRYPTED + "\n";

	for (int i = 0; i < size; i++) delete[] bytes[i];
	delete[] bytes;
}

string loadFile(string path)
{
	// funkcja odpowiedzialna za wczytanie pliku i zwrócenie go jako string
	ifstream myFile;
	string retFile;
	string buffer;
	myFile.open(path);
	do
	{
		getline(myFile, buffer);
		retFile = retFile + buffer + "\n";
	} while (!(myFile.eof()));
	myFile.close();
	return retFile;
}

void convertToBinary(unsigned int letter, unsigned int tab[])
{
	// konwersja inta na postaæ binarn¹ w formie tablicy zer i jedynek
	unsigned int number = letter;

	for (int i = 7; i >= 0; i--)
	{
		tab[i] = number % 2;
		number /= 2;
	}
}