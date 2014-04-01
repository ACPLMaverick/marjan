
#include <iostream>
#include <cmath>
#define GNUPLOT_PATH "C:\\gnuplot\\bin"
#include "gnuplot_i.hpp"
using namespace std;

//f(x) = cos(x/2), x (0,6), x0=pi

double potega(double x, int y)
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
	 return wartosc < 0;
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
	return sin(x);
}

double cosax(double x)
{
	return cos(pow(2, x));
}

double acosx(double x)
{
	return pow(2, cos(x)) - 1;
}
double sinax(double x)
{
	return 0.5*sin(0.25*x);
}
double wykl(double x)
{
	return pow(2, cos(3 * x)) - 1;	//musia³em zostawiæ pow, bo obs³uguje double
}
double wielomian(double x)
{
	// schemat hornera
	// ustalam wspolczynniki:
	double wspolczynniki[] = { 4.0, -8.0, 0.0, 2.0, 0.25 };
	int stopien = 4;
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
	Gnuplot myPlot;

	int mode, iter, function, stopien;
	double a, b, e;
	double (*wybranaFunkcja)(double);

	cout << "Wybierz funkcje:" << endl
		<< "1: 4x^4 - 8x^3 + 2x + 0.25" << endl
		<< "2: cos(0.5x)" << endl
		<< "3: 0.5*sin(x/4)" << endl
		<< "4: 2^(cos(3x))-1" << endl
		<< "5: 2^(cos(x)) - 1" << endl;
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
		wybranaFunkcja = wielomian;
		myPlot.set_title("4x^4 - 8x^3 + 2x + 0.25");
	}
	else if (function == 2)
	{
		wybranaFunkcja = cosx;
		myPlot.set_title("cos(0.5x)");
	}
	else if (function == 3)
	{
		wybranaFunkcja = sinax;
		myPlot.set_title("0.5sin(x/4)");
	}
	else if (function == 4)
	{
		wybranaFunkcja = wykl;
		myPlot.set_title("2^(cos(3x))-1");
	}
	else if (function == 5)
	{
		wybranaFunkcja = acosx;
		myPlot.set_title("2^(cos(x))-1");
	}
	else
	{
		cout << "Blad!" << endl;
		exit(1);
	}
	
	system("CLS");

	// sprawdzanie warunku o ró¿nych znakach na krañcach przedzia³u

	if (wybranaFunkcja(a)*wybranaFunkcja(b) > 0)
	{
		cout << "W podanym przedziale nie ma miejsca zerowego - znaki funkcji na krancach przedzialu nie sa rozne." << endl;
		system("Pause");
		exit(1);
	}

	//

	double wynik_bisekcja;
	double wynik_falsi;

	if (e < 0)
	{
		wynik_bisekcja = bisekcja_iter(wybranaFunkcja, a, b, e, iter);
		wynik_falsi = falsi_iter(wybranaFunkcja, a, b, e, iter);
		cout << "METODA BISEKCJI: " << wynik_bisekcja << endl;
		cout << "METODA FALSI: " << wynik_falsi << endl;
	}
	else
	{
		wynik_bisekcja = bisekcja_dokladnosc(wybranaFunkcja, a, b, e);
		wynik_falsi = falsi_dokladnosc(wybranaFunkcja, a, b, e);
		cout << "METODA BISEKCJI DOKLADNOSC: " << wynik_bisekcja << endl;
		cout << "METODA FALSI DOKLADNOSC: " << wynik_falsi << endl;
	}

	// wykres
	myPlot.set_xlabel("X");
	myPlot.set_ylabel("Y");
	myPlot.set_style("lines");
	myPlot.set_grid();
	myPlot.set_xrange(a, b);
	double zakres = 100 * (abs(b - a));
	
	vector<double> x(zakres);
	vector<double> y(zakres);
	vector<double> x_wyniki_b(1);
	vector<double> y_wyniki_b(1);
	vector<double> x_wyniki_f(1);
	vector<double> y_wyniki_f(1);

	x_wyniki_b[0] = wynik_bisekcja;
	x_wyniki_f[0] = wynik_falsi;
	y_wyniki_b[0] = wybranaFunkcja(wynik_bisekcja);
	y_wyniki_f[0] = wybranaFunkcja(wynik_falsi);

	for (double i = 0.0; i < zakres; i = i + 1.0)
	{
		if(a*b >= 0) x[i] = a + b*(i / zakres);
		else x[i] = a + 2 * b*(i / zakres);
		y[i] = wybranaFunkcja(x[i]);
	}

	myPlot.plot_xy(x, y, "funkcja");

	myPlot.set_style("points");
	myPlot.set_pointsize(2.0);

	myPlot.plot_xy(x_wyniki_b, y_wyniki_b, "x0 - bisekcja");
	myPlot.plot_xy(x_wyniki_f, y_wyniki_f, "x0 - falsi");

	system("PAUSE");
	return 0;
}

