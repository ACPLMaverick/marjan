#include "okno.h"
#include "ui_okno.h"

Okno::Okno(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::Okno)
{
    ui->setupUi(this);
}

Okno::~Okno()
{
    delete ui;
}
