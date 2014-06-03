// Zadanie05.cpp : Defines the entry point for the console application.
//
//TODO: B£¥D APROKSYMACJI

#include <iostream>
#define GNUPLOT_PATH "C:\\gnuplot\\bin"
#define _CRT_SECURE_NO_WARNINGS
#include "gnuplot_i.hpp"
#include "Funkcje.h"

using namespace std;

int f = podajFunkcje();
int stopien = podaj("stopien wielomianu");
int wezly = podaj("ilosc wezlow");
double **macierz = wypelnij(wezly);

double(*func[])(double) = { exponent, wielomian, modul, liniowa, cos };

double potega(int a, int n)
{
	return 0;
}

void wyczyscWektor(vector<double> &w)
{
	for (int i = 0; i < w.size(); i++)
	{
		w[i] = 0;
	}
}

void aproksymacja(double A, vector<double> &xApr, vector<double> &yApr, int stopien, double(*func)(double))
{
	vector<double> lambdy(stopien + 1);
	vector<double> wspolczynniki(stopien + 1);
	double blad = 0;
	for (int i = 0; i <= stopien; i++)							//Liczenie wspó³czynników lambda
	{
		wyczyscWektor(wspolczynniki);
		wyznaczWspolczynniki(wspolczynniki, i);
		lambdy[i] = laguerre(func, wspolczynniki, i, wezly, macierz);
		cout << lambdy[i] << endl;
	}

	cout << endl;

	vector<double> wspol_apr(stopien + 1);						//Liczenie wspó³czynników wielomianiu aproksymuj¹cego
	for (int i = 0; i < stopien + 1; i++)
	{
		for (int j = 0; j < stopien + 1; j++)
		{
			wyczyscWektor(wspolczynniki);
			wyznaczWspolczynniki(wspolczynniki, j);
			wspol_apr[i] += lambdy[j] * wspolczynniki[i];
		}
	}

	for (double x = 0; x <= A; x += 0.1)						//obliczanie wartoœci wielomianu aproksymuj¹cego
	{
		xApr.push_back(x);
		double y = 0;
		y = horner(wspol_apr, stopien, x);
		cout << "x= " << x << " y= " << y << endl;
		yApr.push_back(y);
	}

	for (int k = 0; k < stopien + 1; k++)						//obliczanie b³êdu aproksymacji ze wzoru laguerre_blad
	{
		cout << "wielomian aprox[" << k << "]= " << wspol_apr[k] << endl;
	}
	blad = laguerre_blad(func, wspol_apr, stopien, wezly, macierz);
	cout << "Blad aproksymacji wynosi: " << blad << endl;
}

int main(int argc, char* argv[])
{
	double a = podaj("kraniec przedzialu aproksymacji");

	//GNUPLOT
	Gnuplot::set_GNUPlotPath(GNUPLOT_PATH);
	Gnuplot myPlot;

	myPlot.set_title("Aproksymacja");
	myPlot.set_xlabel("X");
	myPlot.set_ylabel("Y");

	myPlot.set_style("lines");
	myPlot.set_grid();
	myPlot.set_xrange(0, a);
	double zasieg_t = abs(a);
	double zasieg = 100 * abs(a);

	vector<double> x_func(zasieg);
	vector<double> x_inter;
	vector<double> y_func(zasieg);
	vector<double> y_inter;
	vector<double> x_node;
	vector<double> y_node;

	for (double i = 0.0; i < zasieg; i += 1.0)
	{
		x_func[i] = a*(zasieg - i) / zasieg;
		y_func[i] = func[f-1](x_func[i]);							//przyk³adowa funkcja
	}

	aproksymacja(a, x_inter, y_inter, stopien, func[f-1]);

	myPlot.plot_xy(x_func, y_func, "Funkcja wejsciowa");
	myPlot.plot_xy(x_inter, y_inter, "Wielomian aproksymacyjny");

	myPlot.set_style("points");
	myPlot.set_pointsize(2.0);

	for (int i = 0; i < wezly; i++)
	{
		delete[] macierz[i];
	}
	delete[] macierz;

	system("PAUSE");
	return 0;
}

