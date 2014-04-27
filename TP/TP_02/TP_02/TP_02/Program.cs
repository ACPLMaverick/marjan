using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TP_01
{
    class Program
    {
        static void Main(string[] args)
        {
            System.Console.WriteLine("Hello world!");

            RentalOffice myOffice = new RentalOffice();
            Generate10 Generator10 = new Generate10(myOffice);
            Generate10K Generator10K = new Generate10K(myOffice);
            myOffice = Generator10.generateCollection();

            /*
            myOffice.showReaders();
            myOffice.showBooks();
            myOffice.showRents();
             */
            
            myOffice.showAll();
            System.Console.ReadKey();
        }
    }
}
