using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Zad1_CSharp
{
    public class Program
    {
        static void Main(string[] args)
        {
            Controller contr = new Controller();

            contr.Initialize();

            while (contr.Run()) ;

            System.Console.WriteLine("Program terminated. Press any key to exit.");
            System.Console.ReadKey();
        }
    }
}
