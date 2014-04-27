using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TP_01
{
    public class Reader
    {
        private string name;
        private string secondName;
        private double pesel;
        public bool hasBook
        {
            get;
            set;
        }

        public Reader(string name, string secondName, double pesel)
        {
            this.name = name;
            this.secondName = secondName;
            this.pesel = pesel;
            hasBook = false;
        }

        // overrided method ToString()
        public override String ToString()
        {
            return "NAME: " + name + ", SECOND NAME: " + secondName + ", PESEL: " + pesel;
        }
    }
}
