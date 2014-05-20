// Zadanie04.cpp : Defines the entry point for the console application.
//
//TODO: MODU£Y

#include <iostream>
#include "Funkcje.h"

using namespace std;

int main(int argc, char* argv[])
{
	double(*simpson[])(double) = { wielomian, cosinus, cosinus2 };
	double(*gauss[])(double) = { horner, cos_modul, cos2 };
	int f;
	cout << "///CA£KOWANIE NUMERYCZNE///" << endl;
	cout << "Wybierz funkcje: \n" <<
		"1. 8x^3 + 2x + 0.25\n" <<
		"2. cos(|x|)\n" <<
		"3. cos(0.5x)\n";
	cin >> f;

	cout << "///KWADRATURA NEWTONA-COTESA///" << endl;
	double a = podaj("kraniec przedzialu calkowania");
	double delta = podaj("delte");
	double dokladnosc = podaj("dokladnosc calkowania");
	double wynik = simpson_granica(simpson[f - 1], a, delta, dokladnosc);
	cout << wynik << endl;

	cout << "///KWADRATURA GAUSSA///" << endl;
	int n = podaj("ilosc wezlow (2-5):");
	double **macierz = new double *[n];				//macierz wêz³ów i wag dla wielomianiu Laguerre'a
	for (int it = 0; it < n; it++)
	{
		macierz[it] = new double[2];
	}

	if (n >= 2 && n <= 5)
	{
		macierz = wypelnij(n);
		for (int i = 0; i < n; i++)
		{
			for (int j = 0; j < 2; j++)
			{
				cout << "macierz[" << i << "][" << j << "]=" << macierz[i][j] << endl;
			}
			cout << endl;
		}
	}

	wynik = laguerre(macierz, gauss[f-1], n);
	cout << wynik << endl;

	for (int i = 0; i < n; i++)
	{
		delete[] macierz[i];
	}
	delete[] macierz;

	system("PAUSE");
	return 0;
}
