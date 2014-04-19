#ifndef KONWERSJA_H
#define KONWERSJA_H

#include <istream>
#include <ostream>

using namespace std;

void encode(istream &in, ostream &out);
int decode(istream &in, ostream &out);

#endif // KONWERSJA_H
