#include "Funkcje.h"

using namespace std;

//INTERFEJS
double podaj(char* i)
{
	double wartosc;
	cout << "Podaj " << i << " : " << endl;
	cin >> wartosc;
	return wartosc;
}

double podajFunkcje()
{
	double wartosc;
	cout << "Wybierz funkcje: \n" <<
		"1. e^(-2*x)\n" <<
		"2. 8x^3 + 2x + 0.25\n";
	cin >> wartosc;
	return wartosc;
}

//FUNKCJE MATEMATYCZNE
unsigned long long silnia(int n)
{
	unsigned long long wynik = 1;
	for (int i = 2; i <= n; i++)
	{
		wynik *= i;
	}
	return wynik;
}

unsigned long long dwumianNewtona(double n, double k)
{
	double wynik = 1;
	if (k == 0 || k == n)
		return (unsigned long long)wynik;
	else
	{
		for (unsigned int i = 1; i <= k; i++)
		{
			wynik *= (n - i + 1) / i;
		}
		return (unsigned long long)wynik;
	}
}

//FUNKCJE WEJŒCIOWE
double horner(double x)
{
	double wspolczynniki[] = { 8.0, 0.0, 2.0, 0.25 };
	int stopien = 3;
	double wynik = wspolczynniki[0];
	for (int i = 1; i < stopien + 1; i++)
	{
		wynik = wynik*x + wspolczynniki[i];
	}

	return wynik;
}

double exponent(double x)
{
	return exp(-2 * x);
}

//FUNKCJE WYKONUJ¥CE ZA£O¯ENIA PROGRAMU
double* wyznaczWspolczynniki(int stopien)
{
	double* wspolczynniki = new double[stopien + 1];
	for (int i = 0; i < stopien + 1; i++)
	{
		wspolczynniki[i] = (pow(-1, i)*dwumianNewtona(stopien, i) / silnia(i)) * silnia(stopien);
		if (stopien % 2 == 1)
			wspolczynniki[i] *= -1;
	}
	return wspolczynniki;
}

double obliczWielomian(int stopien, double x)
{
	double* wspolczynniki = new double[stopien + 1];
	wspolczynniki = wyznaczWspolczynniki(stopien);
	double wynik = wspolczynniki[stopien];
	for (int i = stopien - 1; i >= 0; i--)
	{
		wynik = wynik*x + wspolczynniki[i];
	}
	delete wspolczynniki;
	return wynik;
}

double funkcja(double(*func)(double), int k, double x)
{
	return obliczWielomian(k, x)*func(x)*exp(-x);
}

double simpson_iter(double(*func)(double), int k, double a, double b, double dokladnosc)
{
	double wynik = 0;
	double wynik_temp = 0;
	double h = 0;
	int n = 1;				//iloœæ podprzedzia³ów
	int i = 0;
	do
	{
		if (wynik_temp != 0)
		{
			wynik = wynik_temp;
		}
		n *= 2;
		if (n < 2 || n % 2 == 1)
		{
			cout << "Zla ilosc podprzedzialow";
			return 0;
		}
		h = abs((b - a) / n);			//odleg³oœæ miêdzy wêz³ami po utworzeniu podprzedzia³ów
		wynik_temp = funkcja(func, k, a) + funkcja(func, k, b);

		for (int i = 1; i < n - 1; i++)					//z³o¿ona postaæ wzoru Simpsona
		{
			if (i % 2 == 0)
			{
				wynik_temp += 4 * funkcja(func, k, a + i*h);		//argumeny funkcji przesuwane o wielokrotnoœæ odleg³oœci
			}
			else
			{
				wynik_temp += 2 * funkcja(func, k, a + i*h);
			}
		}
		wynik_temp *= h / 3;
		i++;
	} while (abs(wynik_temp - wynik) > dokladnosc);
	cout << "warunek:" << abs(wynik_temp - wynik) << endl;
	cout << "Liczba iteracji: " << i << endl;
	return wynik;
}

double simpson_granica(double(*func)(double), int k, double a, double delta, double dokladnosc)
{
	double wynik = simpson_iter(func, k, 0, a, dokladnosc);
	double wynik_temp = dokladnosc;
	int i = 0;
	while (wynik_temp >= dokladnosc)
	{
		wynik_temp = simpson_iter(func, k, a, a + delta, dokladnosc);
		a += delta;
		wynik += wynik_temp;
		i++;
	}
	return wynik;
}
