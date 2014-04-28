using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TP_01
{
    public class Generate : IGenerate
    {
        public RentalOffice generatedOffice;
        private char[] chars = new char[58];
        private Random rnd = new Random();

        public Generate()
        {
            for (int i = 0; i < 58; i++)
            {
                chars[i] = (char)(i + 65);
            }
        }

        public Generate(RentalOffice sentOffice)
        {
            this.generatedOffice = sentOffice;

            for (int i = 0; i < 58; i++)
            {
                chars[i] = (char)(i + 65);
            }
        }

        public String createString(int length)
        {
            String myString = "";
            for(int i=0; i<length; i++)
            {
                myString += chars[rnd.Next(0, 47)];
            }
            return myString;
        }

        public Book createBook(int key)
        {
            return new Book(createString(5), createString(10), 2014, key);
        }

        public Reader createReader()
        {
            return new Reader(createString(7), createString(14), 01010101010);
        }

        public Rent createRent(Book book, Reader reader)
        {
            return new Rent(book, reader);
        }

        public virtual RentalOffice generateCollection() { return null; }
    }
}
