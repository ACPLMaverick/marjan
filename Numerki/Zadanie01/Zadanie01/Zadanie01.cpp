
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

double sprawdz(double (*func)(double), double x1)	//funkcja, która w za³o¿eniu mia³a wczeœniej koñczyæ algorytm
{
	return func(x1);
}

bool sprawdzPrzedzial(double (*func)(double), double x1, double a)
{
	double wartosc = func(x1)*func(a);
	if (wartosc < 0)
	{
		return true;
	}
	else return false;
}

double modul(double (*func)(double), double x1)
{
	return abs(func(x1));
	//return 1.0;
}

// METODA BISEKCJI - funkcja, pocz¹tek przedzia³u, koniec przedzia³u, epsilon, iloœc iteracji
double bisekcja_iter(double(*func)(double), double a, double b, double e, int count)
{
	double temp_x1 = miejsceZerowe(a, b);

	if (count > 0 && sprawdz(func, temp_x1)!=0)	
	{
		if (sprawdzPrzedzial(func, temp_x1, a) == 0)
			temp_x1 = bisekcja_iter(func, temp_x1, b, e, count - 1);
		else
			temp_x1 = bisekcja_iter(func, a, temp_x1, e, count - 1);
	}
	return temp_x1;
}

double bisekcja_dokladnosc(double(*func)(double), double a, double b, double e)
{
	double temp_x1;
	do
	{
		temp_x1 = miejsceZerowe(a, b);
		if (sprawdzPrzedzial(func, temp_x1, a))
		{
			b = temp_x1;
		}
		else
		{
			a = temp_x1;
		}
		//cout << e << "   " << func(temp_x1) << endl;
	} while (abs(func(temp_x1))>e);
	return temp_x1;
}

//METODA REGULA FALSI
double cieciwa(double (*func)(double), double a, double b)
{
	return (a*func(b) - b*func(a)) / (func(b) - func(a));
}

// funkcja, pocz¹tek przedzia³u, koniec przedzia³u, epsilon, iloœc iteracji
double falsi_iter(double(*func)(double), double a, double b, double e, int count)
{
	double temp_x1 = cieciwa(func, a, b);

	if (count > 0 && sprawdz(func, temp_x1)!=0)
	{
		if (sprawdzPrzedzial(func, temp_x1, a) <= 0)
			temp_x1 = falsi_iter(func, temp_x1, a, e, count - 1);
		else
			temp_x1 = falsi_iter(func, temp_x1, b, e, count - 1);
	}
	return temp_x1;
}

double falsi_dokladnosc(double (*func)(double), double a, double b, double e)
{
	double temp_x1;
	do
	{
		temp_x1 = cieciwa(func, a, b);
		if (sprawdzPrzedzial(func, temp_x1, a))
		{
			b = temp_x1;
		}
		else
		{
			a = temp_x1;
		}
		//cout << e << "   " << func(temp_x1) << endl;
	} while (abs(func(temp_x1))>e);
	return temp_x1;
}

/////////////////////////////////////////////////////
//		funkcje
double cosx(double x)
{
	return cos(0.5*x);
}

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
	return pow(2, cos(3 * x)) - 1;	//musia³em zostawiæ pow, bo obs³uguje double
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
	//Wybierany jest ten przedzia³, dla którego spe³nione jest drugie za³o¿enie, tzn. albo f(x_{1})f(a)<0 albo f(x_{1})f(b)<0. 
	//Ca³y proces powtarzany jest dla wybranego przedzia³u.
	Gnuplot::set_GNUPlotPath(GNUPLOT_PATH);

	int mode, iter, function, stopien;
	double a, b, e;
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
		wybranaFunkcja = cosx;
	}
	else if (function == 3)
	{
		wybranaFunkcja = sinax;
	}
	else if (function == 4)
	{
		wybranaFunkcja = wykl;
	}
	else if (function == 5)
	{
		wybranaFunkcja = acosx;
	}
	else
	{
		cout << "Blad!" << endl;
		exit(1);
	}
	
	system("CLS");

	if (e < 0)
	{
		cout << "METODA BISEKCJI: " << bisekcja_iter(wybranaFunkcja, a, b, e, iter) << endl;
		cout << "METODA FALSI: " << falsi_iter(wybranaFunkcja, a, b, e, iter) << endl;
	}
	else
	{
		cout << "METODA BISEKCJI DOKLADNOSC: " << bisekcja_dokladnosc(wybranaFunkcja, a, b, e) << endl;
		cout << "METODA FALSI DOKLADNOSC: " << falsi_dokladnosc(wybranaFunkcja, a, b, e) << endl;
	}


	system("PAUSE");
	return 0;
}

