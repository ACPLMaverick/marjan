using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TP_04.Model
{
    class GenerateManual : Generate
    {
        public GenerateManual(RentalOffice myOffice)
        {
            this.generatedOffice = myOffice;
        }
        public override RentalOffice generateCollection()
        {
            generatedOffice.put(new Book("Archipelag Gulag","Aleksandr Solzenicyn",1973,1));
            generatedOffice.put(new Book("Zycie i los", "Wasilij Grossman", 1959, 2));
            generatedOffice.put(new Book("Dzieci Arbatu", "Anatolij Rybakow", 1989, 3));
            generatedOffice.put(new Book("Akwarium", "Wiktor Suworow", 1985, 4));
            generatedOffice.put(new Book("Lodolamacz", "Wiktor Suworow", 1987, 5));

            generatedOffice.put(new Reader("Jan", "Kowalski", 12345));
            generatedOffice.put(new Reader("Adam", "Nowak", 23456));
            generatedOffice.put(new Reader("Zenon", "Rybinski", 98234));
            generatedOffice.put(new Reader("Jan", "Rudnicki", 13456));

            generatedOffice.put(new Rent(generatedOffice.getBook(1), generatedOffice.getReader(0)));
            generatedOffice.put(new Rent(generatedOffice.getBook(2), generatedOffice.getReader(0)));
            generatedOffice.put(new Rent(generatedOffice.getBook(3), generatedOffice.getReader(3)));
            generatedOffice.put(new Rent(generatedOffice.getBook(4), generatedOffice.getReader(1)));
            generatedOffice.put(new Rent(generatedOffice.getBook(2), generatedOffice.getReader(0)));
            return generatedOffice;
        }
    }
}
