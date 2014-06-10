#include"Funkcje.h"

using namespace std;

//INTERFEJS
double podaj(char* i)
{
	double wartosc;
	cout << "Podaj " << i << " : " << endl;
	cin >> wartosc;
	return wartosc;
}

double** stworzMacierz(int n)
{
	double** F = new double*[n];
	for (int i = 0; i < n; i++)
	{
		F[i] = new double[n];
	}
	return F;
}

//FUNKCJE MATEMATYCZNE
double licz_e1(double t)				//e_1(t) = E_m*sin(omega*t + beta)
{
	return E_m * sin(omega*t + beta);
}

double licz_U_mfe(double fi)			//wzór 5.164
{
	return (l_fe / a2)*sinh(fi / (a1 * S_fe));
}

double licz_i2(double i1, double fi)	//wzór 5.167
{
	return (i1*w1 - fi*R_delta - licz_U_mfe(fi)) / w2;
}

double licz_R_mfe(double fi)			//wzór 5.168a
{
	return (l_fe / a1*a2*S_fe)*cosh(fi / a1*S_fe);
}

double detM(double fi)					//wyznacznik macierzy
{
	return L_r1*(L_r2*((R_delta + licz_R_mfe(fi)) / w2) + w2) + (w1*w1*L_r2) / w2;
}

//RÓWNANIA UK£ADU RÓWNAÑ
double F0(double t, double i1, double fi, double i0, double u_c)
{
	return ((L_r2*((R_delta + licz_R_mfe(fi)) / w2) + w2) * (licz_e1(t) - R1*i1) - w1*(u_c + R2*licz_i2(i1, fi))) / detM(fi);
}

double F1(double t, double i1, double fi, double i0, double u_c)
{
	return ((w1 / w2)*L_r2*(licz_e1(t) - R1*i1) + L_r1*(u_c + R2*licz_i2(i1, fi))) / detM(fi);
}

double F2(double t, double i1, double fi, double i0, double u_c)
{
	return (u_c - R0*i0) / L0;
}

double F3(double t, double i1, double fi, double i0, double u_c)
{
	return (licz_i2(i1, fi) - i0) / C0;
}