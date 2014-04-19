#include <QtGui/QApplication>
#include "okno.h"
#include <cstdlib>

#include <fstream>
#include <iostream>

#include "conversions.h"

using namespace std;
/*
fstream *my_open(char *file)
{
    ifstream str("")
}

fstream *my_close(char *file);*/

int main(int argc, char *argv[]) // w tej tab 0 element to nazwa programu a kolejne argumenty , int argc to liczba argumentow
{
    cout << argc << " " << argv[1] << ", " << argv[2];
    if (argc == 4) {
        ifstream in;
        ofstream out;
        in.open(argv[2]);
        out.open(argv[3]);
        if (strcmp(argv[1], "decode") == 0) {
            cout << "enc";
            decode(in, out);
        } else if (strcmp(argv[1], "encode") == 0) {
            cout << "dec";
            encode(in, out);
        } //else if (strcmp(argv[1], ""));
        return 0;
    }

    QApplication a(argc, argv);
    Okno w;
    w.show();

    return a.exec();
}
