#ifndef FUNKCJE_H
#define FUNKCJE_H
#define _USE_MATH_DEFINES //do u¿ycia Pi

#include<iostream>
#include<vector>
#include<math.h>

using namespace std;

//STA£E MATEMATYCZNE
const double a1 = 0.238829;
const double a2 = 2.0596744;
const double E_m = 380 * sqrt(2);
const double omega = 100 * M_PI;
const double beta = M_PI / 4;
const double w1 = 114;
const double w2 = 1890;
const double teta = w1 / w2;
const double l_fe = 1.77;
const double S_fe = 0.011005;
const double delta = 0.4*pow(10, -3);
const double mi0 = (0.4*M_PI)*pow(10, -6);
const double R_delta = delta / (mi0*S_fe);
const double R0 = 2000;
const double R1 = 0.5;
const double R2 = R1 / (teta*teta);
const double L0 = 2000 / omega;
const double L_r1 = 0.5 / omega;
const double L_r2 = L_r1 * teta*teta;
const double C0 = 1 / (omega * 5000);

//INTERFEJS
double podaj(char* i);
double** stworzMacierz(int n);

//FUNKCJE MATEMATYCZNE
double licz_e1(double t);				//e_1(t) = E_m*sin(omega*t + beta)
double licz_U_mfe(double fi);			//wzór 5.164
double licz_i2(double i1, double fi);	//wzór 5.167
double licz_R_mfe(double fi);			//wzór 5.168A
double detM(double fi);					//wyznacznik macierzy

//RÓWNANIA UK£ADU RÓWNAÑ
double F0(double t, double i1, double fi, double i0, double u_c);
double F1(double t, double i1, double fi, double i0, double u_c);
double F2(double t, double i1, double fi, double i0, double u_c);
double F3(double t, double i1, double fi, double i0, double u_c);

#endif FUNKCJE_H