#ifndef OKNO_H
#define OKNO_H

#include <QMainWindow>

namespace Ui {
    class Okno;
}

class Okno : public QMainWindow
{
    Q_OBJECT

public:
    explicit Okno(QWidget *parent = 0);
    ~Okno();

private:
    Ui::Okno *ui;
};

#endif // OKNO_H
