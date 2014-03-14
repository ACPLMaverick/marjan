using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TP_01
{
    class Czytelnik
    {
        private string imie;
        private string nazwisko;
        private double pesel;
        public bool ma_ksiazke
        {
            get;
            set;
        }

        public Czytelnik(string imie, string nazwisko, double pesel)
        {
            this.imie = imie;
            this.nazwisko = nazwisko;
            this.pesel = pesel;
            ma_ksiazke = false;
        }
    }
}
