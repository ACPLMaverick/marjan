#include "Funkcje.h"

using namespace std;

double** wypelnij(int n)
{
	double **macierz = new double *[n];				//macierz wêz³ów i wag dla wielomianiu Laguerre'a
	for (int it = 0; it < n; it++)
	{
		macierz[it] = new double[2];
	}
	switch (n)
	{
	case 2:
	{
			  macierz[0][0] = 0.585786; macierz[0][1] = 0.853553;
			  macierz[1][0] = 3.414214; macierz[1][1] = 0.146447;
			  return macierz;
	}
	case 3:
	{
			  macierz[0][0] = 0.415775; macierz[0][1] = 0.711093;
			  macierz[1][0] = 2.294280; macierz[1][1] = 0.278518;
			  macierz[2][0] = 6.289945; macierz[2][1] = 0.010389;
			  return macierz;
	}
	case 4:
	{
			macierz[0][0] = 0.322548; macierz[0][1] = 0.603154;
			macierz[1][0] = 1.745761; macierz[1][1] = 0.357419;
			macierz[2][0] = 4.536620; macierz[2][1] = 0.038888;
			macierz[3][0] = 9.395070; macierz[3][1] = 0.000539;
			return macierz;
	}
	case 5:
	{
			macierz[0][0] = 0.263560; macierz[0][1] = 0.521756;
			macierz[1][0] = 1.413403; macierz[1][1] = 0.398667;
			macierz[2][0] = 3.596426; macierz[2][1] = 0.075942;
			macierz[3][0] = 7.085810; macierz[3][1] = 0.003612;
			macierz[4][0] = 12.640801; macierz[4][1] = 0.000032;
			return macierz;
	}
	}
}

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
		"2. 8x^3 +4x + 0.25\n" <<
		"3. |x|\n" <<
		"4. 2x + 5\n" << 
		"5. cos(x)\n";
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

//FUNKCJE WEJŒCIOWE
double horner(vector<double> wspolczynniki, int stopien, double x)
{
	double wynik = wspolczynniki[stopien];
	for (int i = stopien - 1; i >= 0; i--)
	{
		wynik = wynik*x + wspolczynniki[i];
	}

	return wynik;
}

double wielomian(double x)
{
	vector<double> wspolczynniki = { 0.25, 4.0, 0.0, 8.0 };
	return horner(wspolczynniki, 3, x);
}

double exponent(double x)
{
	return exp(-0.1 * x);
}

double modul(double x)
{
	return abs(x);
}

double liniowa(double x)
{
	vector<double> wspolczynniki = { 5.0, 2.0 };
	return horner(wspolczynniki, 1, x);
}

//FUNKCJE WYKONUJ¥CE ZA£O¯ENIA PROGRAMU
void wyznaczWspolczynniki(vector<double> &w, int stopien)	//wyznaczanie wspó³czynników wielomianu aproksymuj¹cego
{
	double y = 1;
	if (stopien == 0) w[0] = 1;
	for (int i = 0; i < stopien+1 ; i++)
	{
		y = potega(-1, i) * dwumianNewtona(stopien, i) / silnia(i);
		w[i] = y;
	}
}

double iloczyn(double(*func)(double), vector<double> w, int stopien, double x) //wzór funkcji potrzebnej do obliczenia wspó³czynnika lambda
{
	return func(x)*horner(w, stopien, x);
}

double roznica(double(*func)(double), vector<double> w, int stopien, double x) //wzór do obliczenia b³êdu aproksymacji
{
	return abs((func(x) - horner(w, stopien, x)));
}

double laguerre(double(*func)(double), vector<double> w, int stopien, int wezly, double** macierz) //algorytm ca³kowania wzorem Gaussa-Laguerre'a
{
	double y = 0.0;
	for (size_t i = 0; i < wezly; ++i) y += macierz[i][1] * iloczyn(func, w, stopien, macierz[i][0]);
	return y;
}

double laguerre_blad(double(*func)(double), vector<double> w, int stopien, int wezly, double** macierz) 
{
	double y = 0.0;
	for (size_t i = 0; i < wezly; ++i) y += macierz[i][1] * roznica(func, w, stopien, macierz[i][0]);	
	return y;
}
