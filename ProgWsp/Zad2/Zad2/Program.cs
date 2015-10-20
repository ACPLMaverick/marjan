using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Zad2
{
    class Program
    {
        static void Main(string[] args)
        {
            Controller contr = new Controller();

            contr.Run();

            Console.WriteLine("Program terminated. Press any key to exit.");
            Console.ReadKey();
        }
    }
}
