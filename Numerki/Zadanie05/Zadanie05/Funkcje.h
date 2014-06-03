#ifndef Funkcje_h
#define Funkcje_h

#include <iostream>
#include <vector>

double** wypelnij(int n);

//INTERFEJS
double podaj(char* i);
double podajFunkcje();

//FUNKCJE MATEMATYCZNE
unsigned long long silnia(int n);
unsigned long long dwumianNewtona(double n, double k);

//FUNKCJE WEJŒCIOWE
double horner(std::vector<double> wspolczynniki, int stopien, double x);
double wielomian(double x);
double exponent(double x);
double modul(double x);
double liniowa(double x);

//FUNKCJE WYKONUJ¥CE ZA£O¯ENIA PROGRAMU
void wyznaczWspolczynniki(std::vector<double> &w, int stopien);
double iloczyn(double(*func)(double), std::vector<double> w, int stopien, double x);
double roznica(double(*func)(double), std::vector<double> w, int stopien, double x);
double laguerre(double(*func)(double), std::vector<double> w, int stopien, int wezly, double** macierz);
double laguerre_blad(double(*func)(double), std::vector<double> w, int stopien, int wezly, double** macierz);

#endif