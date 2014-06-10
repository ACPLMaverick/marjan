using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TP_04.Model
{
    public class ModelController
    {
        private RentalOffice myOffice;
        private Generate generatorM;
        private ObservableCollection<Book> propBook;
        private ObservableCollection<Reader> propReader;
        private ObservableCollection<Rent> propRent;

        public ModelController()
        {
            generatorM = new GenerateManual(new RentalOffice());
            myOffice = generatorM.generateCollection();
            propBook = null;
            propReader = null;
            propRent = null;
        }

        public void setObsCols(ObservableCollection<Book> propBook, ObservableCollection<Reader> propReader, ObservableCollection<Rent> propRent)
        {
            this.propBook = propBook;
            this.propReader = propReader;
            this.propRent = propRent;

            propBook.CollectionChanged += ChangeBook;
            propReader.CollectionChanged += ChangeReader;
            propRent.CollectionChanged += ChangeRent;
        }

        public RentalOffice GetOffice()
        {
            return myOffice;
        }

        public void SerializeOffice(string path)
        {
            Converter converter = new ConverterMARJAN(path);
            converter.Serialize(myOffice);
        }

        public void DeserializeOffice(string path)
        {
            Converter converter = new ConverterMARJAN(path);
            myOffice = converter.Deserialize();
        }

        private void ChangeBook(object sender, EventArgs e)
        {
            ObservableCollection<Book> sndCol = (ObservableCollection<Book>)sender;
            SerializableList<Book> myList = new SerializableList<Book>();
            foreach (Book element in sndCol)
            {
                myList.Add(element);
            }
            myOffice.books = myList;
        }

        private void ChangeReader(object sender, EventArgs e)
        {
            ObservableCollection<Reader> sndCol = (ObservableCollection<Reader>)sender;
            SerializableList<Reader> myList = new SerializableList<Reader>();
            foreach (Reader element in sndCol)
            {
                myList.Add(element);
            }
            myOffice.readers = myList;
        }

        private void ChangeRent(object sender, EventArgs e)
        {
            ObservableCollection<Rent> sndCol = (ObservableCollection<Rent>)sender;
            EventObservableCollection<Rent> myList = new EventObservableCollection<Rent>();
            foreach (Rent element in sndCol)
            {
                myList.Add(element);
            }
            myOffice.rents = myList;
        }
    }
}
