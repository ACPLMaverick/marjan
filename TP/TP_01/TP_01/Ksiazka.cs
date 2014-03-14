using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TP_01
{
    class Ksiazka
    {
        private string tytul;
        private string autor;
        private double rok_wydania;
        public int klucz;
        public bool wypozyczona
        {
            get;
            set;
        }

        public Ksiazka(string tytul, string autor, double rok_wydania)
        {
            this.tytul = tytul;
            this.autor = autor;
            this.rok_wydania = rok_wydania;
            wypozyczona = false;
        }
    }
}
