using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TP_01
{
    public class Rent
    {
        public Book book
        {
            get;
            set;
        }
        public Reader reader
        {
            get;
            set;
        }
        public Rent(Book book, Reader reader)
        {
            this.book = book;
            this.reader = reader;
            this.reader.hasBook = true;
            this.book.rented = true;
        }

        // overrided method ToString()
        public override String ToString()
        {
            return "\nBOOK: " + book.ToString() + "\nREADER: " + reader.ToString();
        }
    }
}
