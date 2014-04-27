using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TP_01
{
    interface IGenerate
    {
        String createString(int length);
        Book createBook(int key);
        Reader createReader();
        Rent createRent(Book book, Reader reader);
        RentalOffice generateCollection();
    }
}
