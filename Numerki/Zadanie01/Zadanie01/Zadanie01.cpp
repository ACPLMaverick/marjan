
#include <iostream>
#include <cmath>
#include "gnuplot_i.hpp"
using namespace std;
#define GNUPLOT_PATH "C:\gnuplot\bin"
//f(x) = cos(x/2), x (0,6), x0=pi

double miejsceZerowe(double a, double b)
{
	return (a + b) / 2;
}

bool sprawdz(double (*func)(double), double n, double x1)
{
	double wartosc = func(n*x1);
	if (wartosc = 0)
	{
		return true;
	}
	else return false;
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

double bisekcja_iter(double a, double b, int count)
{
	double temp_x1 = miejsceZerowe(a, b);

	if (count > 0)
	{
		if (sprawdzPrzedzial(cos, 0.5, temp_x1, a) == 0)
			temp_x1 = bisekcja_iter(temp_x1, b, count - 1);
		else
			temp_x1 = bisekcja_iter(a, temp_x1, count - 1);
	}
	return temp_x1;
}

void bisekcja_dokladnosc()
{

}

int main(int argc, char* argv[])
{
	//Wybierany jest ten przedzia³, dla którego spe³nione jest drugie za³o¿enie, tzn. albo f(x_{1})f(a)<0 albo f(x_{1})f(b)<0. 
	//Ca³y proces powtarzany jest dla wybranego przedzia³u.
	Gnuplot::set_GNUPlotPath(GNUPLOT_PATH);
	int iter;
	cout << "Podaj ilosc iteracji: ";
	cin >> iter;
	//bisekcja_iter(0, 6, iter);

	
	cout << endl << bisekcja_iter(0, 6, iter) << endl;
	system("PAUSE");
	return 0;
}

