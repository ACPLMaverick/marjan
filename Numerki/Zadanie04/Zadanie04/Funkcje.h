
#include <iostream>

using namespace std;

double podaj(char* i)
{
	double wartosc;
	cout << "Podaj " << i << " : " << endl;
	cin >> wartosc;
	return wartosc;
}

double** wypelnij(int n)
{
	double **macierz = new double *[n];				//macierz wêz³ów i wag dla wielomianiu Laguerre'a
	for (int it = 0; it < n; it++)
	{
		macierz[it] = new double[2];
	}
	if (n == 2)
	{
		macierz[0][0] = 0.585786; macierz[0][1] = 0.853553;
		macierz[1][0] = 3.414214; macierz[1][1] = 0.146447;
		return macierz;
	}
	else if (n == 3)
	{
		macierz[0][0] = 0.415775; macierz[0][1] = 0.711093;
		macierz[1][0] = 2.294280; macierz[1][1] = 0.278518;
		macierz[2][0] = 6.289945; macierz[2][1] = 0.010389;
		return macierz;
	}
	else if (n == 4)
	{
		macierz[0][0] = 0.322548; macierz[0][1] = 0.603154;
		macierz[1][0] = 1.745761; macierz[1][1] = 0.357419;
		macierz[2][0] = 4.536620; macierz[2][1] = 0.038888;
		macierz[3][0] = 9.395070; macierz[3][1] = 0.000539;
		return macierz;
	}
	else if (n == 5)
	{
		macierz[0][0] = 0.263560; macierz[0][1] = 0.521756;
		macierz[1][0] = 1.413403; macierz[1][1] = 0.398667;
		macierz[2][0] = 3.596426; macierz[2][1] = 0.075942;
		macierz[3][0] = 7.085810; macierz[3][1] = 0.003612;
		macierz[4][0] = 12.640801; macierz[4][1] = 0.000032;
		return macierz;
	}
}

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

//Funkcje f(x) przyjmowane przez kwadraturê Gaussa
double cos_modul(double x)
{
	return cos(abs(x));
}

double cos2(double x)
{
	return cos(0.5 * x);
}

//Funkcje postaci w(x)*f(x) przyjmowane przez wzór Simpsona
//gdzie w(x) jest funkcj¹ wagow¹ - dla nas e^(-x)
double wielomian(double x)
{
	return exp(-x)*horner(x);
}

double cosinus(double x)
{
	return exp(-x)*cos_modul(x);
}

double cosinus2(double x)
{
	return exp(-x)*cos2(x);
}

//Funkcje licz¹ce ca³ki obiema metodami
double simpson_iter(double(*func)(double), double a, double b, double dokladnosc)
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
		wynik_temp = func(a) + func(b);

		for (int i = 1; i < n - 1; i++)					//z³o¿ona postaæ wzoru Simpsona
		{
			if (i % 2 == 0)
			{
				wynik_temp += 4 * func(a + i*h);		//argumeny funkcji przesuwane o wielokrotnoœæ odleg³oœci
			}
			else
			{
				wynik_temp += 2 * func(a + i*h);
			}
		}
		wynik_temp *= h / 3;
		i++;
	} while (abs(wynik_temp - wynik) > dokladnosc);
	cout << "Liczba iteracji [simpson_iter]: " << i << endl;
	return wynik;
}

double simpson_granica(double(*func)(double), double a, double delta, double dokladnosc)
{
	double wynik = simpson_iter(func, 0, a, dokladnosc);
	double wynik_temp = dokladnosc;
	int i = 0;
	while (wynik_temp >= dokladnosc)
	{
		wynik_temp = simpson_iter(func, a, a + delta, dokladnosc);
		a += delta;
		wynik += wynik_temp;
		i++;
	}
	return wynik;
}

double laguerre(double** macierz, double(*func)(double), double a)
{
	double suma = 0;
	for (int j = 0; j < a; j++)
	{
		suma += macierz[j][1] * func(macierz[j][0]);
	}
	return suma;
}