// Zadanie06.cpp : Defines the entry point for the console application.
//

using namespace std;

#define GNUPLOT_PATH "C:\\gnuplot\\bin"
#define _CRT_SECURE_NO_WARNINGS

#include "Funkcje.h"
#include "gnuplot_i.hpp"

//ZMIENNE W UK£ADZIE RÓWNAÑ RÓ¯NICZKOWYCH
vector<double> x(4);
vector<double> x2(4);
vector<double> i1_t(1); 						    //x[0] = i1(t); x[1] = fi(t); x[2] = i0(t); x[3] = u_c(t)
vector<double> fi_t(1);
vector<double> i0_t(1);
vector<double> u_c_t(1);
vector<double> i1_t2(1); 						    //x[0] = i1(t); x[1] = fi(t); x[2] = i0(t); x[3] = u_c(t)
vector<double> fi_t2(1);
vector<double> i0_t2(1);
vector<double> u_c_t2(1);
double** F = stworzMacierz(4);						//kolejne równania ró¿niczkowe

//FUNKCJE WYKONUJ¥CE ZA£O¯ENIA PROGRAMU
void K(double t, double i1, double fi, double i0, double u_c, double h, vector<double> &x, vector<double> &i1_t, vector<double> &fi_t, vector<double> &i0_t, vector<double> &u_c_t)
{
	//cout << "t: " << t << endl;
	//LICZENIE K1
	F[0][0] = F0(t, i1, fi, i0, u_c);
	F[1][0] = F1(t, i1, fi, i0, u_c);
	F[2][0] = F2(t, i1, fi, i0, u_c);
	F[3][0] = F3(t, i1, fi, i0, u_c);
	for (int i = 0; i < 4; i++)
	{
		F[i][0] *= h;
	}

	//LICZENIE K2
	F[0][1] = F0(t + 0.5*h, i1 + 0.5*F[0][0], fi + 0.5*F[1][0], i0 + 0.5*F[2][0], u_c + 0.5*F[3][0]);
	F[1][1] = F1(t + 0.5*h, i1 + 0.5*F[0][0], fi + 0.5*F[1][0], i0 + 0.5*F[2][0], u_c + 0.5*F[3][0]);
	F[2][1] = F2(t + 0.5*h, i1 + 0.5*F[0][0], fi + 0.5*F[1][0], i0 + 0.5*F[2][0], u_c + 0.5*F[3][0]);
	F[3][1] = F3(t + 0.5*h, i1 + 0.5*F[0][0], fi + 0.5*F[1][0], i0 + 0.5*F[2][0], u_c + 0.5*F[3][0]);
	for (int i = 0; i < 4; i++)
	{
		F[i][1] *= h;
	}

	//LICZENIE K3
	F[0][2] = F0(t + 0.5*h, i1 + 0.5*F[0][1], fi + 0.5*F[1][1], i0 + 0.5*F[2][1], u_c + 0.5*F[3][1]);
	F[1][2] = F1(t + 0.5*h, i1 + 0.5*F[0][1], fi + 0.5*F[1][1], i0 + 0.5*F[2][1], u_c + 0.5*F[3][1]);
	F[2][2] = F2(t + 0.5*h, i1 + 0.5*F[0][1], fi + 0.5*F[1][1], i0 + 0.5*F[2][1], u_c + 0.5*F[3][1]);
	F[3][2] = F3(t + 0.5*h, i1 + 0.5*F[0][1], fi + 0.5*F[1][1], i0 + 0.5*F[2][1], u_c + 0.5*F[3][1]);
	for (int i = 0; i < 4; i++)
	{
		F[i][2] *= h;
	}

	//LICZENIE K4
	F[0][3] = F0(t + h, i1 + F[0][2], fi + F[1][2], i0 + F[2][2], u_c + F[3][2]);
	F[1][3] = F1(t + h, i1 + F[0][2], fi + F[1][2], i0 + F[2][2], u_c + F[3][2]);
	F[2][3] = F2(t + h, i1 + F[0][2], fi + F[1][2], i0 + F[2][2], u_c + F[3][2]);
	F[3][3] = F3(t + h, i1 + F[0][2], fi + F[1][2], i0 + F[2][2], u_c + F[3][2]);
	for (int i = 0; i < 4; i++)
	{
		F[i][3] *= h;
	}

	//LICZENIE y(t + h)
	double tmp0, tmp1, tmp2, tmp3;
	tmp0 = x[0] + (F[0][0] + 2 * F[0][1] + 2 * F[0][2] + F[0][3]) / 6;
	tmp1 = x[1] + (F[1][0] + 2 * F[1][1] + 2 * F[1][2] + F[1][3]) / 6;
	tmp2 = x[2] + (F[2][0] + 2 * F[2][1] + 2 * F[2][2] + F[2][3]) / 6;
	tmp3 = x[3] + (F[3][0] + 2 * F[3][1] + 2 * F[3][2] + F[3][3]) / 6;

	x[0] = tmp0;
	x[1] = tmp1;
	x[2] = tmp2;
	x[3] = tmp3;

	i1_t.push_back(x[0]);
	fi_t.push_back(x[1]);
	i0_t.push_back(x[2]);
	u_c_t.push_back(x[3]);
}

void K2(double t, double i1, double fi, double i0, double u_c, double h, vector<double> &x2, vector<double> &i1_t2, vector<double> &fi_t2, vector<double> &i0_t2, vector<double> &u_c_t2)
{
	//cout << "t: " << t << endl;
	//LICZENIE K1
	F[0][0] = F0(t, i1, fi, i0, u_c);	//K11
	F[1][0] = F1(t, i1, fi, i0, u_c);	//K12
	F[2][0] = F2(t, i1, fi, i0, u_c);	//K13
	F[3][0] = F3(t, i1, fi, i0, u_c);	//K14
	for (int i = 0; i < 4; i++)
	{
		F[i][0] *= h;
	}

	//LICZENIE K2
	F[0][1] = F0(t + 0.4*h, i1 + 0.4*F[0][0], fi + 0.4*F[1][0], i0 + 0.4*F[2][0], u_c + 0.4*F[3][0]);	//K21
	F[1][1] = F1(t + 0.4*h, i1 + 0.4*F[0][0], fi + 0.4*F[1][0], i0 + 0.4*F[2][0], u_c + 0.4*F[3][0]);	//K22
	F[2][1] = F2(t + 0.4*h, i1 + 0.4*F[0][0], fi + 0.4*F[1][0], i0 + 0.4*F[2][0], u_c + 0.4*F[3][0]);	//K23
	F[3][1] = F3(t + 0.4*h, i1 + 0.4*F[0][0], fi + 0.4*F[1][0], i0 + 0.4*F[2][0], u_c + 0.4*F[3][0]);	//K24
	for (int i = 0; i < 4; i++)
	{
		F[i][1] *= h;
	}

	//LICZENIE K3
	F[0][2] = F0(t + 0.45573726*h, i1 + 0.29697760*F[0][0] + 0.15875966*F[0][1], fi + 0.29697760*F[1][0] + 0.15875966*F[1][1], i0 + 0.29697760*F[2][0] + 0.15875966*F[2][1], u_c + 0.29697760*F[3][0] + 0.15875966*F[3][1]);
	F[1][2] = F1(t + 0.45573726*h, i1 + 0.29697760*F[0][0] + 0.15875966*F[0][1], fi + 0.29697760*F[1][0] + 0.15875966*F[1][1], i0 + 0.29697760*F[2][0] + 0.15875966*F[2][1], u_c + 0.29697760*F[3][0] + 0.15875966*F[3][1]);
	F[2][2] = F2(t + 0.45573726*h, i1 + 0.29697760*F[0][0] + 0.15875966*F[0][1], fi + 0.29697760*F[1][0] + 0.15875966*F[1][1], i0 + 0.29697760*F[2][0] + 0.15875966*F[2][1], u_c + 0.29697760*F[3][0] + 0.15875966*F[3][1]);
	F[3][2] = F3(t + 0.45573726*h, i1 + 0.29697760*F[0][0] + 0.15875966*F[0][1], fi + 0.29697760*F[1][0] + 0.15875966*F[1][1], i0 + 0.29697760*F[2][0] + 0.15875966*F[2][1], u_c + 0.29697760*F[3][0] + 0.15875966*F[3][1]);
	for (int i = 0; i < 4; i++)
	{
		F[i][2] *= h;
	}

	//LICZENIE K4
	F[0][3] = F0(t + h, i1 + 0.21810038*F[0][0] - 3.05096470*F[0][1] + 3.83286432*F[0][2], fi + 0.21810038*F[1][0] - 3.05096470*F[1][1] + 3.83286432*F[1][2], i0 + 0.21810038*F[2][0] - 3.05096470*F[2][1] + 3.83286432*F[2][2], u_c + +0.21810038*F[3][0] - 3.05096470*F[3][1] + 3.83286432*F[3][2]);
	F[1][3] = F1(t + h, i1 + 0.21810038*F[0][0] - 3.05096470*F[0][1] + 3.83286432*F[0][2], fi + 0.21810038*F[1][0] - 3.05096470*F[1][1] + 3.83286432*F[1][2], i0 + 0.21810038*F[2][0] - 3.05096470*F[2][1] + 3.83286432*F[2][2], u_c + +0.21810038*F[3][0] - 3.05096470*F[3][1] + 3.83286432*F[3][2]);
	F[2][3] = F2(t + h, i1 + 0.21810038*F[0][0] - 3.05096470*F[0][1] + 3.83286432*F[0][2], fi + 0.21810038*F[1][0] - 3.05096470*F[1][1] + 3.83286432*F[1][2], i0 + 0.21810038*F[2][0] - 3.05096470*F[2][1] + 3.83286432*F[2][2], u_c + +0.21810038*F[3][0] - 3.05096470*F[3][1] + 3.83286432*F[3][2]);
	F[3][3] = F3(t + h, i1 + 0.21810038*F[0][0] - 3.05096470*F[0][1] + 3.83286432*F[0][2], fi + 0.21810038*F[1][0] - 3.05096470*F[1][1] + 3.83286432*F[1][2], i0 + 0.21810038*F[2][0] - 3.05096470*F[2][1] + 3.83286432*F[2][2], u_c + +0.21810038*F[3][0] - 3.05096470*F[3][1] + 3.83286432*F[3][2]);
	for (int i = 0; i < 4; i++)
	{
		F[i][3] *= h;
	}

	//LICZENIE y(t + h)
	double tmp0, tmp1, tmp2, tmp3;
	tmp0 = x2[0] + 0.17476028*F[0][0] - 0.55148053* F[0][1] + 1.20553547 * F[0][2] + 0.17118478 * F[0][3];
	tmp1 = x2[1] + 0.17476028*F[1][0] - 0.55148053* F[1][1] + 1.20553547 * F[1][2] + 0.17118478 * F[1][3];
	tmp2 = x2[2] + 0.17476028*F[2][0] - 0.55148053* F[2][1] + 1.20553547 * F[2][2] + 0.17118478 * F[2][3];
	tmp3 = x2[3] + 0.17476028*F[3][0] - 0.55148053* F[3][1] + 1.20553547 * F[3][2] + 0.17118478 * F[3][3];

	x2[0] = tmp0;
	x2[1] = tmp1;
	x2[2] = tmp2;
	x2[3] = tmp3;

	i1_t2.push_back(x2[0]);
	fi_t2.push_back(x2[1]);
	i0_t2.push_back(x2[2]);
	u_c_t2.push_back(x2[3]);
}


int main(int argc, char* argv[])
{
	int n;
	int podzialka = 512;
	double t1 = 0.0;
	double t2 = 0.2;
	double h = (t2 - t1) / podzialka;			//krok

	for (double t = t1; t <= t2; t += h)
	{
		K(t, i1_t[0], fi_t[0], i0_t[0], u_c_t[0], h, x, i1_t, fi_t, i0_t, u_c_t);
		K2(t, i1_t2[0], fi_t2[0], i0_t2[0], u_c_t2[0], h, x2, i1_t2, fi_t2, i0_t2, u_c_t2);
	}

	cout << "Podaj wykres do wyswietlenia: \n" <<
		"1. I_1(t)\n" <<
		"2. fi(t)\n" <<
		"3. I_0(t)\n" <<
		"4. U_c(t)\n";

	cin >> n;

	//GNUPLOT
	Gnuplot::set_GNUPlotPath(GNUPLOT_PATH);
	Gnuplot myPlot;

	myPlot.set_title("Uk³ad równañ ró¿niczkowych");
	myPlot.set_xlabel("t");
	myPlot.set_ylabel("y");

	myPlot.set_style("lines");
	myPlot.set_grid();
	myPlot.set_xrange(0.0, 0.2);
	double zasieg = podzialka + 2;

	vector<double> x_func(zasieg);
	vector<double> x_func2(zasieg);
	vector<double> y_func(zasieg);
	vector<double> y_func2(zasieg);
	vector<double> x_node;
	vector<double> y_node;

	switch (n)
	{
	case 1: 	for (double i = 0; i < i1_t.size(); i++)
				{
					x_func[i] = 0.2*(zasieg - i) / zasieg;
					x_func2[i] = 0.2*(zasieg - i) / zasieg;
				}
				for (double i = 0; i < i1_t.size(); i++)
				{
					y_func[i] = i1_t[i];							
					y_func2[i] = i1_t2[i];
				}
				break;
	case 2:		for (double i = 0; i < fi_t.size(); i++)
				{
					x_func[i] = 0.2*(zasieg - i) / zasieg;
					x_func2[i] = 0.2*(zasieg - i) / zasieg;
				}
				for (double i = 0; i < fi_t.size(); i++)
				{
					y_func[i] = fi_t[i];	
					y_func2[i] = fi_t2[i];
				}
				break;
	case 3:		for (double i = 0; i < i0_t.size(); i++)
				{
					x_func[i] = 0.2*(zasieg - i) / zasieg;
					x_func2[i] = 0.2*(zasieg - i) / zasieg;
				}
				for (double i = 0; i < i0_t.size(); i++)
				{
					y_func[i] = i0_t[i];	
					y_func2[i] = i0_t2[i];
				}
	case 4:		for (double i = 0; i < u_c_t.size(); i++)
				{
					x_func[i] = 0.2*(zasieg - i) / zasieg;
					x_func2[i] = 0.2*(zasieg - i) / zasieg;
				}
				for (double i = 0; i < u_c_t.size(); i++)
				{
					y_func[i] = u_c_t[i];	
					y_func2[i] = u_c_t2[i];
				}
				break;
	}

	myPlot.plot_xy(x_func, y_func, "Funkcja wejsciowa");
	myPlot.plot_xy(x_func2, y_func2, "Funkcja wejsciowa 2");

	myPlot.set_style("points");
	myPlot.set_pointsize(2.0);

	system("PAUSE");
	return 0;
}

