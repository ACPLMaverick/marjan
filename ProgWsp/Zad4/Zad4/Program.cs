using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Zad4
{
    class Program
    {
        static void Main(string[] args)
        {
            //strukturalnie
            //Compare the Need <= Available of first Process
            //If true then Calculate Available = Available + Allocation of that process
            //If false then compare the Need with the next Process
            //Process is repeat top to bottom order until all the process will be compared

            //wielowątkowo
            //lock na pobieraniu zasobów
            //sprawdza które zasoby może pożyczyć
            //po jakimś czasie (Thread.Sleep) zwraca zasoby
            //jeśli wykorzystał już swoje życzenia - zakończ proces

            Controller contr = new Controller();

            contr.Run();

            Console.WriteLine("Program terminated. Press any key to exit.");
            Console.ReadKey();
        }
    }
}
