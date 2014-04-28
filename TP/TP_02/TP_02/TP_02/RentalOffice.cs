using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TP_01
{
    public class RentalOffice : IRentalOffice<Reader>, IRentalOffice<Book>, IRentalOffice<Rent>
    {
        // collections
        private List<Reader> readers;
        private Dictionary<int, Book> books;
        private EventObservableCollection<Rent> rents;
        ////////////

        // constructor
        public RentalOffice()
        {
            readers = new List<Reader>();
            books = new Dictionary<int, Book>();
            rents = new EventObservableCollection<Rent>();
            rents.Added += new AddedEventHandler(RentsAdded);
        }

        //eventhandler
        private void RentsAdded(object sender, EventArgs e)
        {
            Console.WriteLine("=== NEW RENT ADDED ===");
        }

        // printing methods
        public void showReaders()
        {
            for (int i = 0; i < readers.Count; i++)
            {
                System.Console.Write(readers[i].ToString() + "\n");
            }
        }

        public void showBooks()
        {
            foreach (KeyValuePair<int, Book> book in books)
            {
                System.Console.Write(book.Value.ToString() + "\n");
            }
        }

        public void showRents()
        {
            for (int i = 0; i < rents.Count; i++)
            {
                System.Console.Write(rents[i].ToString() + "\n");
            }
        }

        // overall printing method
        public void showAll()
        {
            System.Console.Write("#####################################\n");
            System.Console.Write("############ CZYTELNICY #############\n");
            foreach (Reader reader in readers)
            {
                System.Console.Write("#####################################\n");
                System.Console.Write(reader.ToString() + "\n");
                System.Console.Write("\n" + "MY BOOKS: \n");

                foreach(Rent rent in rents)
                {
                    if(rent.reader == reader)
                    {
                        System.Console.Write("  " + rent.book.ToString() + "\n");
                    }
                }
            }

            System.Console.Write("\n#####################################\n");
            System.Console.Write("############## KSIAZKI ##############\n");
            System.Console.Write("#####################################\n");

            foreach (KeyValuePair<int, Book> book in books)
            {
                System.Console.Write(book.Key + ": " + book.Value.ToString() + "\n");
            }
        }

        // data access methods
        // Implementation of DataHandler<T> interface
        public void put(Reader reader)
        {
            readers.Add(reader);
        }

        public void put(Book book)
        {
            books.Add(book.key, book);
        }

        public void put(Rent rent)
        {
            rents.Add(rent);
        }

        public Reader getReader(int place)
        {
            return readers[place];
        }

        public Book getBook(int key)
        {
            return books[key];
        }

        public Rent getRent(int plac)
        {
            return rents[plac];
        }

        public int count(string collectionType)
        {
            switch(collectionType)
            {
                case "READERS":
                    return readers.Count;
                    break;
                case "BOOKS":
                    return books.Count;
                    break;
                case "RENTS":
                    return rents.Count;
                    break;
                default:
                    return -1;
                    break;
            }
        }
        //////////////////

        // events
    }
}


