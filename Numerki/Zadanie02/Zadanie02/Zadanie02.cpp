// Zadanie02.cpp : Defines the entry point for the console application.
//

#include <iostream>
#include <iomanip>
#include <cmath>
#include <fstream>
#include <vector>

using namespace std;

const long double epsilon = 0.000001;

int podaj(char* i)									//funkcja dla podawania wielkoœci macierzy
{
	int wartosc;
	cout << "Podaj wartosc " << i << " : " << endl;
	cin >> wartosc;
	return wartosc;
}

void wyswietl(double **macierz, int m)		//wyswietlanie zawartoœci macierzy
{
	for (int i = 0; i < m; i++)
	{
		cout << "| ";
		for (int j = 0; j < m+1; j++)
		{
			cout << macierz[i][j] << " ";
		}
		cout << "|\n";
	}
}

void zamien(double **macierz, int m, int i)	//zamiana wierszy (i - wiersz do zamiany)
{
	for (int k = i+1; k < m; k++)
	{
		if (macierz[k][i] != 0)				// macierz[k][i] > macierz[i][i]
		{
			double temp;
			for (int l = 0; l < m+1; l++)
			{
				temp = macierz[k][l];
				macierz[k][l] = macierz[i][l];
				macierz[i][l] = temp;
			}
		}
	}
}

bool eliminuj(double **macierz, int m)		//etap eliminacji zmiennych algorytmu 
{
	double mnoznik = 0;
	for (int i = 0; i < m-1; i++)
	{
		if (abs(macierz[i][i]) < epsilon)
		{
			zamien(macierz, m, i);
		}
		for (int j = i+1; j < m; j++)
		{
			mnoznik = -macierz[j][i] / macierz[i][i];
			for (int k = i+1; k <= m; k++)
			{
				macierz[j][k] += mnoznik * macierz[i][k];
			}
		}
	}
	return true;
}

bool oblicz(double **macierz, double *wyniki, int m)	//etap wyznaczania niewiadomych postêpowaniem odwrotnym
{
	double wyrazWolny = 0;
	for (int i = m - 1; i >= 0; i--)
	{
		if (abs(macierz[i][i]) < epsilon && macierz[i][i] != macierz[i][m])
		{
			cout << "Uklad sprzeczny" << endl;
			return false;
		}
		else if (abs(macierz[i][i]) < epsilon && macierz[i][i] == macierz[i][m])
		{
			cout << "Uklad nieoznaczony" << endl;
			return false;
		}
		wyrazWolny = macierz[i][m];
		for (int j = m - 1; j > i; j--)
		{
			wyrazWolny -= macierz[i][j] * wyniki[j];
		}
		wyniki[i] = wyrazWolny / macierz[i][i];
	}
	return true;
}

int main(int argc, char* argv[])
{
	ifstream file;
	int m = podaj("m");

	double *wyniki = new double[m];

	//dynamiczne tworzenie macierzy
	double **macierz = new double * [m];
	for (int it = 0; it < m; it++)
	{
		macierz[it] = new double[m+1];
	}

	//obs³uga plików
	file.open("liczby.txt", ios::in | ios::out);
	if (file.good())
	{
		cout << "Uzyskano dostep do pliku" << endl;
		//operacje na pliku
		while (!file.eof())
		{
			for (int i = 0; i < m; i++)
			{
				for (int j = 0; j < m+1; j++)
				{
					file >> macierz[i][j];
				}
			}
		}
		wyswietl(macierz, m);
		file.close();
	}
	else cout << "Dostep do pliku zabroniony" << endl;

	if (eliminuj(macierz, m) && oblicz(macierz, wyniki, m))
	{
		for (int i = 0; i < m; i++)
		{
			cout << "x" << i+1 << " = "  << wyniki[i] << endl;
		}
	}
	else
	{
		cout << "Operacja nie powiodla sie" << endl;
	}

	for (int i = 0; i < m; i++)
	{
		delete[] macierz[i];
	}
	delete[] macierz;
	delete[] wyniki;

	system("PAUSE");
	return 0;
}
