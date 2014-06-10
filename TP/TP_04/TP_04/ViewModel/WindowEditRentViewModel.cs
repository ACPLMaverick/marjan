using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;

namespace TP_04.ViewModel
{
    public class WindowEditRentViewModel : Common.BindableBase
    {
        private Model.ModelController modelController;
        private ObservableCollection<Model.Rent> prop;
        private int selectedItem;
        public ObservableCollection<Model.Book> prop_books {get; set;}
        public ObservableCollection<Model.Reader> prop_readers { get; set; }

        public int prevSelectedBook { get; set; }
        public int prevSelectedReader { get; set; }

        private Model.Rent _rent;
        public Model.Rent rent
        {
            get { return _rent; }
            set
            {
                if(_rent != value)
                {
                    _rent = value;
                    OnPropertyChanged("book");
                }
            }
        }

        public Common.DelegateCommand ButtonOKCommand { get; set; }
        public Common.DelegateCommand ButtonCancelCommand { get; set; }

        public WindowEditRentViewModel()
        {
            ButtonOKCommand = new Common.DelegateCommand(ButtonOKClicked);
            ButtonCancelCommand = new Common.DelegateCommand(ButtonCancelClicked);
            this.modelController = null;
            this.prop = null;
            this.prop_books = null;
            this.selectedItem = -1;
        }

        public WindowEditRentViewModel(Model.ModelController controller, ObservableCollection<Model.Rent> prop, ObservableCollection<Model.Book> prop_books, ObservableCollection<Model.Reader> prop_readers, int selectedItem)
        {
            ButtonOKCommand = new Common.DelegateCommand(ButtonOKClicked);
            ButtonCancelCommand = new Common.DelegateCommand(ButtonCancelClicked);
            this.modelController = controller;
            this.prop = prop;
            this.prop_books = prop_books;
            this.prop_readers = prop_readers;
            this.selectedItem = selectedItem;
            this.rent = prop.ElementAt(selectedItem);
            setPreviousValues();
        }

        public void ButtonOKClicked(object sender)
        {
            List<object> sentList = (List<object>)sender;
            ComboBox comboBooks = (ComboBox)sentList[0];
            ComboBox comboReaders = (ComboBox)sentList[1];
            int selectedBook = comboBooks.SelectedIndex;
            int selectedReader = comboReaders.SelectedIndex;
            editRent(prop_books.ElementAt(selectedBook), prop_readers.ElementAt(selectedReader));
            GetActiveWindow().Close();
        }

        public void ButtonCancelClicked(object sender)
        {
            GetActiveWindow().Close();
        }

        private void editRent(Model.Book book, Model.Reader reader)
        {
            if(rent.book.Equals(book))
            {
                rent.book = book;
            }
            else
            {
                rent.book.rented -= 1;
                book.rented += 1;
                book.wasRented = true;
                rent.book = book;
            }

            if(rent.reader.Equals(reader))
            {
                rent.reader = reader;
            }
            else
            {
                rent.reader.rented -= 1;
                reader.rented += 1;
                rent.reader = reader;
            }
        }

        private void setPreviousValues()
        {
            int b = 0, r = 0;
            List<Model.Book> tempBooks = modelController.GetOffice().getBookCollection();
            List<Model.Reader> tempReaders = modelController.GetOffice().getReaderCollection();
            foreach(Model.Book book in tempBooks)
            {
                if (book.Equals(rent.book)) break;
                b++;
            }
            foreach(Model.Reader reader in tempReaders)
            {
                if (reader.Equals(rent.reader)) break;
                r++;
            }
            prevSelectedBook = b;
            prevSelectedReader = r;
        }
    }
}
