using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Runtime.Serialization;
using System.Text;
using System.Threading.Tasks;

namespace TP_04.Model
{
    [Serializable()]
    [System.Xml.Serialization.XmlInclude(typeof(Reader))]
    [System.Xml.Serialization.XmlInclude(typeof(Book))]
    [System.Xml.Serialization.XmlInclude(typeof(Rent))]
    public class RentalOffice : IRentalOffice<Reader>, IRentalOffice<Book>, IRentalOffice<Rent>, ISerializable
    {
        // collections
        public SerializableList<Reader> readers;
        public SerializableList<Book> books;
        public EventObservableCollection<Rent> rents;
        ////////////

        // constructor
        public RentalOffice()
        {
            readers = new SerializableList<Reader>();
            books = new SerializableList<Book>();
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
            foreach (Book book in books)
            {
                System.Console.Write(book.ToString() + "\n");
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

            int it = 1;
            foreach (Book book in books)
            {
                System.Console.Write(it.ToString() + ": " + book.ToString() + "\n");
                it++;
            }
        }

        // data access methods
        // Implementation of DataHandler<T> interface
        public void put(Reader reader) { readers.Add(reader); }

        public void put(Book book) { books.Add(book); }

        public void put(Rent rent) { rents.Add(rent); }

        public Reader getReader(int place) { return readers[place]; }

        public Book getBook(int key) { return books[key]; }

        public Rent getRent(int place) { return rents[place]; }

        public SerializableList<Reader> getReaderCollection() { return readers; }

        public SerializableList<Book> getBookCollection() { return books; }

        public EventObservableCollection<Rent> getRentCollection() { return rents; }

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

        // do serializacji
        public RentalOffice(SerializationInfo info, StreamingContext context)
        {
            //this.Add((T)info.GetValue(typeof(T).Name, typeof(T)));
            this.readers = (SerializableList<Reader>)info.GetValue("readers", typeof(SerializableList<Reader>));
            this.books = (SerializableList<Book>)info.GetValue("books", typeof(SerializableList<Book>));
            this.rents = (EventObservableCollection<Rent>)info.GetValue("rents", typeof(EventObservableCollection<Rent>));
        }

        public void GetObjectData(SerializationInfo info, StreamingContext context)
        {
            info.AddValue("readers", this.readers, typeof(SerializableList<Reader>));
            info.AddValue("books", this.books, typeof(SerializableList<Reader>));
            info.AddValue("rents", this.rents, typeof(EventObservableCollection<Rent>));
        }
    }
}


