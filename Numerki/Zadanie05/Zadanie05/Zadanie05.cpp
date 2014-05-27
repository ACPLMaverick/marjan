// Zadanie05.cpp : Defines the entry point for the console application.
//

#include <iostream>
#define GNUPLOT_PATH "C:\\gnuplot\\bin"
#define _CRT_SECURE_NO_WARNINGS
#include "gnuplot_i.hpp"
#include "Funkcje.h"

using namespace std;

//int f = podajFunkcje();
//int stopien = podaj("stopien wielomianu");
double(*func[])(double) = { exponent, horner };

double potega(int a, int n)
{
	return 0;
}


void aproksymacja(double(*func)(double), double a, double delta, vector<double> &x_apr, vector<double> &y_apr, int stopien)
{
	double* a_k = new double[stopien + 1];
	for (int k = 0; k < stopien + 1; k++)
	{
		//korzystam tu ze wzoru na wspo³czynnik a: ca³ka (od 0 do inf) z iloczynu f(t)*wielomian
														//Laguerre'a_{k}(t)*waga czyli exp^(-t)
		a_k[k] = simpson_granica(func, k, a, delta, 0.001);
		cout << "a_k[" << k << "]=" << a_k[k] << endl;
	}

	for (double x = 0; x <= a; x += 0.1)				
	{
		x_apr.push_back(x);
		double y = 0;
		for (int i = 0; i < stopien + 1; i++)
		{
			y += a_k[i] * obliczWielomian(i, x);       //rozwi¹zywanie wartoœci wielomianu aproksymuj¹cego jako suma wspó³czynnika a i wielomian Laguerre'a_{k}(t)
		}
		y_apr.push_back(y);
	}
}

int main(int argc, char* argv[])
{
	//double a = podaj("kraniec przedzialu aproksymacji");
	//double delta = podaj("delte");

	//cout << "ObliczWielomian L dla k = 0: " << obliczWielomian(0, 2) << endl;
	//cout << "ObliczWielomian L dla k = 1: " << obliczWielomian(1, 2) << endl;
	//cout << "ObliczWielomian L dla k = 2: " << obliczWielomian(2, 2) << endl;
	//cout << "ObliczWielomian L dla k = 3: " << obliczWielomian(3, 2) << endl;

	

	//GNUPLOT
	//Gnuplot::set_GNUPlotPath(GNUPLOT_PATH);
	//Gnuplot myPlot;

	//myPlot.set_title("Aproksymacja");
	//myPlot.set_xlabel("X");
	//myPlot.set_ylabel("Y");

	//myPlot.set_style("lines");
	//myPlot.set_grid();
	//myPlot.set_xrange(0, a);
	//double zasieg_t = abs(a);
	//double zasieg = 100 * abs(a);

	//vector<double> x_func(zasieg);
	//vector<double> x_inter;
	//vector<double> y_func(zasieg);
	//vector<double> y_inter;
	//vector<double> x_node;
	//vector<double> y_node;

	//for (double i = 0.0; i < zasieg; i += 1.0)
	//{
	//	x_func[i] = a*(zasieg - i) / zasieg;
	//	y_func[i] = func[f-1](x_func[i]);							//przyk³adowa funkcja
	//}

	//aproksymacja(func[f-1], a, delta, x_inter, y_inter, stopien);

	//myPlot.plot_xy(x_func, y_func, "Funkcja wejsciowa");
	//myPlot.plot_xy(x_inter, y_inter, "Wielomian aproksymacyjny");

	//myPlot.set_style("points");
	//myPlot.set_pointsize(2.0);

	system("PAUSE");
	return 0;
}

