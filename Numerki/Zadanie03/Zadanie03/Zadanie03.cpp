// Zadanie03.cpp : Defines the entry point for the console application.
//

#include <iostream>
#include <vector>
#include <string>
#define GNUPLOT_PATH "C:\\gnuplot\\bin"
#define _CRT_SECURE_NO_WARNINGS
#include "gnuplot_i.hpp"

using namespace std;

double podaj(char* i)									//funkcja dla podawania wielko�ci macierzy
{
	double wartosc;
	cout << "Podaj " << i << " : " << endl;
	cin >> wartosc;
	return wartosc;
}

double odleglosc(int m, double a, double b)				//obliczanie odleg�o�ci mi�dzy w�z�ami
{
	return (b - a) / (m-1);
}

double oblicz(double(*func)(double), double x)
{
	return func(x);
}

double obliczX(double **macierz, double t, double odleglosc)
{
	return macierz[0][0] + t*odleglosc;
}

vector<double> obliczWspolczynnik(double **macierz, int m)
{
	vector<double> vec;
	for (int n = 0; n < m; n++)
	{
		double suma = 0;
		for (int i = 0; i <= n; i++)
		{
			double mnoz = 1;
			for (int j = 0; j <= n; j++)
			{
				if (j != i)
				{
					mnoz *= (macierz[i][0] - macierz[j][0]);
				}
			}
			suma += (macierz[i][1] / mnoz);
		}
		vec.push_back(suma);
	}
	return vec;
}

double obliczWielomian(vector<double> vec, double **macierz, double t)			//wykorzystano warunek interpolacji
{
	int stopien = vec.size() - 1;
	double iloczyn = 1;
	double wynik = vec.at(0);
	for (int i = 1; i <= stopien; i++)
	{
		iloczyn = 1;
		for (int j = 0; j < i; j++)
		{
			iloczyn *= t - macierz[j][0];
		}
		wynik += vec.at(i)*iloczyn;
	}
	return wynik;
}

void wyswietl(double **macierz, int m)							//pomocnicza metoda wy�wietlaj�ca macierz
{
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < 2; j++)
		{
			cout << macierz[i][j] << " ";
		}
		cout << endl;
	}
}

void baza(int i)												//wyswietlanie bazy wielomian�w
{
	if (i == 0) cout << "";
	else
	{
		for (int j = 0; j < i; j++)
		{
			cout << "*( t - " << j << " )";
		}
	}
}

//wyswietlanie wielomianu na ekran
void wyswietlWielomian(vector<double> vec, int counter)
{
	for (int i = 0; i < counter; i++)
	{
		if (i>0 && i < counter) cout << "+";
		if (vec[i] > 0) cout << vec[i];
		else cout << "(" << vec[i] << ")";
		baza(i);
	}
	cout << endl;
}


// DO WYBIERANIA FUNKCJI!!!!!!!!!!!!!!!

typedef double(*fptr)(double);


double func01(double x)
{
	return x - 5;
}

double func02(double x)
{
	return abs(x);
}

double func03(double x)
{
	// schemat hornera
	// ustalam wspolczynniki:
	double wspolczynniki[] = { 4.0, -8.0, 0.0, 2.0, 0.25 };
	int stopien = 4;
	double wynik = wspolczynniki[0];
	for (int i = 1; i < stopien + 1; i++)
	{
		wynik = wynik*x + wspolczynniki[i];
	}

	return wynik;
}

double func04(double x)
{
	return abs(cos(0.5*x));
}

double func05(double x)
{
	return cos(abs(x));
}

fptr wybierzFunkcje()
{
	double(*funkcja)(double) = nullptr;
	int wybor;
	bool zlyWybor = false;

	cout << "Wybor funkcji: \n"
		<< "1: f(x) = x - 5 \n"
		<< "2: f(x) = |x| \n"
		<< "3: f(x) = 4x^4 - 8x^3 + 2x + 0.25 \n"
		<< "4: f(x) = |cos(0.5x)| \n"
		<< "5: f(x) = cos(|x|) \n";
	
	do
	{
		zlyWybor = false;
		cin >> wybor;
		if (wybor == 1) funkcja = func01;
		else if (wybor == 2) funkcja = func02;
		else if (wybor == 3) funkcja = func03;
		else if (wybor == 4) funkcja = func04;
		else if (wybor == 5) funkcja = func05;
		else zlyWybor = true;
	} while (zlyWybor);

	return funkcja;
}


/////////////////////////////////////////

int main(int argc, char* argv[])
{
	// wyb�r funkcji 
	double(*wybranaFunkcja)(double) = wybierzFunkcje();
	
	int m = (int)podaj("ilosc wezlow");

	//tablica wsp�czynnik�w a_i oraz wsp�czynnik�w ostatecznego wielomianu
	vector<double> wspolczynniki(m);
	vector<double> x;

	//dynamiczne tworzenie macierzy - m jest liczb� w�z��w
	double **macierz = new double *[m];				//ilo�� w�z��w z zerem
	for (int it = 0; it < m; it++)
	{
		macierz[it] = new double[2];				//kolumna x i y
	}
	
	macierz[0][0] = podaj("pierwszy kraniec przedzialu interpolacji");
	macierz[m-1][0] = podaj("drugi kraniec przedzialu interpolacji");
	double a = macierz[0][0];
	double b = macierz[m - 1][0];

	for (int i = 1; i < m-1; i++)						//wypelnianie kolumny x wartosciami wez��w r�wnoodleg�ych
	{
		macierz[i][0] = macierz[i-1][0];
		macierz[i][0] += odleglosc(m, macierz[0][0], macierz[m - 1][0]);
	}
	
	for (int i = 0; i < m; i++)							//wypelnianie kolumny y warto�ciami funkcji w w�z�ach
	{
		macierz[i][1] = oblicz(wybranaFunkcja, macierz[i][0]);
	}

	wyswietl(macierz, m);

	wspolczynniki = obliczWspolczynnik(macierz, m);

	wyswietlWielomian(wspolczynniki, m);

	for (int i = 0; i < m; i++)
	{
		cout << "a" << i << " = " << wspolczynniki[i] << endl;
	}

	for (int i = 0; i < m; i++)
	{
		x.push_back(obliczX(macierz, i, odleglosc(m, macierz[0][0], macierz[m - 1][0])));
	}

	for (int i = 0; i < m; i++)
	{
		cout << "w(" << x[i] << ") = " << obliczWielomian(wspolczynniki, macierz, x[i]) << endl;  //to s� dane do wielomianu interpolacji
	}

	//GNUPLOT
	Gnuplot::set_GNUPlotPath(GNUPLOT_PATH);
	Gnuplot myPlot;

	myPlot.set_title("Interpolacja");
	myPlot.set_xlabel("X");
	myPlot.set_ylabel("Y");

	myPlot.set_style("lines");
	myPlot.set_grid();
	myPlot.set_xrange(macierz[0][0], macierz[m-1][0]);
	double zasieg_t = abs(b - a);
	double zasieg = 100 * abs(b - a);

	vector<double> x_func(zasieg);
	vector<double> t_inter(zasieg);
	vector<double> x_inter(zasieg);
	vector<double> y_func(zasieg);
	vector<double> y_inter(zasieg);
	vector<double> x_node;
	vector<double> y_node;

	for (double i = 0.0; i < zasieg; i += 1.0)
	{
		x_func[i] = a*((zasieg - i) / zasieg) + b*(i / zasieg);
		y_func[i] = oblicz(wybranaFunkcja, x_func[i]);							//przyk�adowa funkcja
	}

	for (double i = 0.0; i < zasieg; i += 1.0)
	{
		x_inter[i] = a*((zasieg - i) / zasieg) + b*(i / zasieg);
		y_inter[i] = obliczWielomian(wspolczynniki, macierz, x_inter[i]);
	}

	// w�z�y
	for (int i = 0; i < m; i++)						
	{
		x_node.push_back(macierz[i][0]);
		y_node.push_back(macierz[i][1]);
	}
	
	myPlot.plot_xy(x_func, y_func, "Funkcja wejsciowa");
	myPlot.plot_xy(x_inter, y_inter, "Wielomian interpolacyjny");

	myPlot.set_style("points");
	myPlot.set_pointsize(2.0);

	myPlot.plot_xy(x_node, y_node, "W�z�y interpolacji");

	//usuwanie macierzy
	for (int i = 0; i < m; i++)
	{
		delete[] macierz[i];
	}
	delete[] macierz;

	system("PAUSE");
	return 0;
}

