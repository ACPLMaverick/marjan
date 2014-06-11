using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TP_04.Model
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
