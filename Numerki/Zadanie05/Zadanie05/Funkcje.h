#ifndef Funkcje_h
#define Funkcje_h

#include <iostream>
#include <vector>

//INTERFEJS
double podaj(char* i);
double podajFunkcje();

//FUNKCJE MATEMATYCZNE
unsigned long long silnia(int n);
unsigned long long dwumianNewtona(double n, double k);

//FUNKCJE WEJŒCIOWE
double horner(double x);
double exponent(double x);

//FUNKCJE WYKONUJ¥CE ZA£O¯ENIA PROGRAMU
double* wyznaczWspolczynniki(int stopien);
double obliczWielomian(int stopien, double x);
double funkcja(double(*func)(double), int k, double x);
double simpson_iter(double(*func)(double), int k, double a, double b, double dokladnosc);
double simpson_granica(double(*func)(double), int k, double a, double delta, double dokladnosc);

#endif