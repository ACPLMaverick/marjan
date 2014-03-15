
#include <iostream>
#include <cmath>
#include "gnuplot_i.hpp"
using namespace std;
#define GNUPLOT_PATH "C:\gnuplot\bin"
//f(x) = cos(x/2), x (0,6), x0=pi

double potega(double x, double y)
{
	if (x == 0 && y <= 0) exit(-1);
	else if (x == 0) return 0;
	else if (x == 1) return 1;
	else
	{
		double n = 1;
		int i;
		if (y < 0)
		{
			x = 1 / x;
			y = -y;
		}
		for (i = 1; i <= y; ++i)
		{
			n *= x;
		}
		return n;
	}
}

//METODA BISEKCJI
double miejsceZerowe(double a, double b)
{
	return (a + b) / 2;
}

double sprawdz(double (*func)(double), double n, double x1)	//funkcja, kt�ra w za�o�eniu mia�a wcze�niej ko�czy� algorytm
{
	return func(n*x1);
}

bool sprawdzPrzedzial(double (*func)(double), double n, double x1, double a)
{
	double wartosc = func(n*x1)*func(n*a);
	if (wartosc < 0)
	{
		return true;
	}
	else return false;
}

double modul(double (*func)(double), double n, double x1)
{
	return abs(func(n*x1));
	//return 1.0;
}

// METODA BISEKCJI - funkcja, wsp�czynnik wewn�trz funkcji, pocz�tek przedzia�u, koniec przedzia�u, epsilon, ilo�c iteracji
double bisekcja_iter(double(*func)(double), double n, double a, double b, double e, int count)
{
	double temp_x1 = miejsceZerowe(a, b);

	if (count > 0 /*|| modul(func, 0.5, temp_x1) > e*/ && sprawdz(func, n, temp_x1)!=0)	
	{
		if (sprawdzPrzedzial(func, n, temp_x1, a) == 0)
			temp_x1 = bisekcja_iter(func, n, temp_x1, b, e, count - 1);
		else
			temp_x1 = bisekcja_iter(func, n, a, temp_x1, e, count - 1);
	}
	return temp_x1;
}

//METODA REGULA FALSI
double cieciwa(double (*func)(double), double n, double a, double b)
{
	return (a*func(n*b) - b*func(n*a)) / (func(n*b) - func(n*a));
}

// funkcja, wsp�czynnik wewn�trz funkcji, pocz�tek przedzia�u, koniec przedzia�u, epsilon, ilo�c iteracji
double falsi_iter(double(*func)(double), double n, double a, double b, double e, int count)
{
	double temp_x1 = cieciwa(func, n, a, b);

	if (count > 0 && sprawdz(func, n, temp_x1)!=0)
	{
		if (sprawdzPrzedzial(func, n, temp_x1, a) <= 0)
			temp_x1 = falsi_iter(func, n, temp_x1, a, e, count - 1);
		else
			temp_x1 = falsi_iter(func, n, temp_x1, b, e, count - 1);
	}
	return temp_x1;
}

/////////////////////////////////////////////////////
//		funkcje
double cosax(double x)
{
	return cos(pow(2, x));
}

double acosx(double x)
{
	return pow(2, cos(x));
}
double sinax(double x)
{
	return 0.5*sin(0.25*x);
}
double wykl(double x)
{
	return pow(2, cos(3 * x)) - 1;	//musia�em zostawi� pow, bo obs�uguje double
}
double wielomian(int wspolczynniki[], int stopien, double x)
{
	// schemat hornera
	double wynik = wspolczynniki[0];
	for (int i = 1; i < stopien; i++)
	{
		wynik = wynik*x + wspolczynniki[i];
	}

	return wynik;
}

/////////////////////////////////////////////////////


//PROGRAM
int main(int argc, char* argv[])
{
	//Wybierany jest ten przedzia�, dla kt�rego spe�nione jest drugie za�o�enie, tzn. albo f(x_{1})f(a)<0 albo f(x_{1})f(b)<0. 
	//Ca�y proces powtarzany jest dla wybranego przedzia�u.
	Gnuplot::set_GNUPlotPath(GNUPLOT_PATH);

	int mode, iter, function, stopien;
	double a, b, e, n;
	double (*wybranaFunkcja)(double);
	double (*wybranaFunkcjaWielomian)(int[], int, double);

	cout << "Wybierz funkcje:" << endl
		<< "1: Wielomian" << endl
		<< "2: cos(0.5x)" << endl
		<< "3: 0.5*sin(x/4)" << endl
		<< "4: 2^(cos(3x))-1" << endl
		<< "5: 2^(cos(x)" << endl;
	cin >> function;

	cout << "Wybierz kryterium zatrzymania: " << endl
		<< "1: |f(x)| < e" << endl
		<< "2: ilosc iteracji" << endl;
	cin >> mode;

	switch (mode)
	{
	case 1:
		cout << "Podaj epsilon: ";
		cin >> e;
		iter = -1;
		break;
	case 2:
		cout << "Podaj ilosc iteracji: ";
		cin >> iter;
		e = -1;
		break;
	default:
		cout << "Blad!" << endl;
		exit(1);
		break;
	}

	cout << "Podaj przedzial:" << endl
		<< "a: ";
	cin >> a;
	cout << "b: ";
	cin >> b;

	if (function == 1)
	{
		/*
		// tworzenie wielomianu
		int stopien, arg;
		cout << endl << "Podaj stopien wielomianu: ";
		cin >> stopien;
		cout << endl;
		int *wspolczynniki = new int[stopien + 1];

		for (int i = 0; i <= stopien; i++)
		{
			cout << "Wspolczynnik przy potedze " << stopien - i << ": ";
			cin >> wspolczynniki[i];
		}
		cout << endl;
		wybranaFunkcjaWielomian = wielomian;

		system("CLS");

		cout << "METODA BISEKCJI: " << bisekcja_iter(wybranaFunkcja, a, b, e, iter) << endl;
		cout << "METODA FALSI: " << falsi_iter(wybranaFunkcja, a, b, e, iter) << endl;
		system("PAUSE");*/
		exit(0);
	}
	else if (function == 2)
	{
		wybranaFunkcja = cos;
		n = 0.5;
	}
	else if (function == 3)
	{
		wybranaFunkcja = sinax;
		n = 1;
	}
	else if (function == 4)
	{
		wybranaFunkcja = wykl;
		n = 1;
	}
	else if (function == 5)
	{
		wybranaFunkcja = acosx;
		n = 1;
	}
	else
	{
		cout << "Blad!" << endl;
		exit(1);
	}
	
	system("CLS");

	cout << "METODA BISEKCJI: " << bisekcja_iter(wybranaFunkcja, n, a, b, e, iter) << endl;
	cout << "METODA FALSI: " << falsi_iter(wybranaFunkcja, n, a, b, e, iter) << endl;

	system("PAUSE");
	return 0;
}

