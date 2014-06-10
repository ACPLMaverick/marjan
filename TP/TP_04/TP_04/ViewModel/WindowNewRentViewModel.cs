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
    public class WindowNewRentViewModel : Common.BindableBase
    {
        private Model.ModelController modelController;
        private ObservableCollection<Model.Rent> prop;
        public ObservableCollection<Model.Book> prop_books {get; set;}
        public ObservableCollection<Model.Reader> prop_readers { get; set; }
        

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

        public WindowNewRentViewModel()
        {
            ButtonOKCommand = new Common.DelegateCommand(ButtonOKClicked);
            ButtonCancelCommand = new Common.DelegateCommand(ButtonCancelClicked);
            this.modelController = null;
            this.prop = null;
            this.prop_books = null;
        }

        public WindowNewRentViewModel(Model.ModelController controller, ObservableCollection<Model.Rent> prop, ObservableCollection<Model.Book> prop_books, ObservableCollection<Model.Reader> prop_readers)
        {
            ButtonOKCommand = new Common.DelegateCommand(ButtonOKClicked);
            ButtonCancelCommand = new Common.DelegateCommand(ButtonCancelClicked);
            this.modelController = controller;
            this.prop = prop;
            this.prop_books = prop_books;
            this.prop_readers = prop_readers;
        }

        public void ButtonOKClicked(object sender)
        {
            List<object> sentList = (List<object>)sender;
            ComboBox comboBooks = (ComboBox)sentList[0];
            ComboBox comboReaders = (ComboBox)sentList[1];
            int selectedBook = comboBooks.SelectedIndex;
            int selectedReader = comboReaders.SelectedIndex;
            createNewRent(modelController.GetOffice().getBookCollection().ElementAt(selectedBook), modelController.GetOffice().getReaderCollection().ElementAt(selectedReader));
            GetActiveWindow().Close();
        }

        public void ButtonCancelClicked(object sender)
        {
            GetActiveWindow().Close();
        }

        private void createNewRent(Model.Book book, Model.Reader reader)
        {
            rent = new Model.Rent(book, reader);
            book.rented += 1;
            book.wasRented = true;
            reader.rented += 1;
            //modelController.GetOffice().getRentCollection().Add(rent);
            prop.Add(rent);
        }
    }
}
