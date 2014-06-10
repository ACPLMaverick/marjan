using Microsoft.Win32;
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
    public class MainWindowViewModel : Common.BindableBase
    {
        private Model.ModelController model;

        private string currentList;

        private ObservableCollection<Model.Book> _books;
        public ObservableCollection<Model.Book> books
        {
            get { return _books; }
            set
            {
                if(_books != value)
                {
                    _books = value;
                    OnPropertyChanged("books");
                }
            }
        }

        private ObservableCollection<Model.Reader> _readers;
        public ObservableCollection<Model.Reader> readers
        {
            get { return _readers; }
            set
            {
                if (_readers != value)
                {
                    _readers = value;
                    OnPropertyChanged("readers");
                }
            }
        }

        private ObservableCollection<Model.Rent> _rents;
        public ObservableCollection<Model.Rent> rents
        {
            get { return _rents; }
            set
            {
                if (_rents != value)
                {
                    _rents = value;
                    OnPropertyChanged("rents");
                }
            }
        }

        public Common.DelegateCommand ButtonExitCommand { get; set; }
        public Common.DelegateCommand ButtonNewCommand { get; set; }
        public Common.DelegateCommand ButtonEditCommand { get; set; }
        public Common.DelegateCommand ButtonDeleteCommand { get; set; }
        public Common.DelegateCommand ButtonLoadCommand { get; set; }
        public Common.DelegateCommand ButtonSaveCommand { get; set; }
        public Common.DelegateCommand MyComboBoxChangedCommand { get; set; }

        public MainWindowViewModel()
        {
            model = new Model.ModelController();

            UpdateLists();

            ButtonExitCommand = new Common.DelegateCommand(ButtonExitClicked);
            ButtonNewCommand = new Common.DelegateCommand(ButtonNewClicked);
            ButtonEditCommand = new Common.DelegateCommand(ButtonEditClicked);
            ButtonDeleteCommand = new Common.DelegateCommand(ButtonDeleteClicked);
            ButtonLoadCommand = new Common.DelegateCommand(ButtonLoadClicked);
            ButtonSaveCommand = new Common.DelegateCommand(ButtonSaveClicked);
            MyComboBoxChangedCommand = new Common.DelegateCommand(MainComboBox_SelectionChanged);

            this.currentList = "books";
        }

        public void ButtonExitClicked(object sender)
        {
            //System.Diagnostics.Debug.WriteLine("button clicked!");
            //Environment.Exit(0);
            System.Windows.Application.Current.Shutdown();
        }

        public void ButtonNewClicked(object sender)
        {
            Window newWindow = null;
            if (currentList == "books")
            {
                newWindow = new WindowNewBook();
                newWindow = (WindowNewBook)newWindow;
                newWindow.DataContext = new WindowNewBookViewModel(model, books);
            }
            else if (currentList == "readers")
            {
                newWindow = new WindowNewReader();
                newWindow = (WindowNewReader)newWindow;
                newWindow.DataContext = new WindowNewReaderViewModel(model, readers);
            }
            else if (currentList == "rents")
            {
                newWindow = new WindowNewRent();
                newWindow = (WindowNewRent)newWindow;
                newWindow.DataContext = new WindowNewRentViewModel(model, rents, books, readers);
            }
            newWindow.Show();
        }

        public void ButtonEditClicked(object sender)
        {
            List<object> array = (List<object>)sender;
            ListView lvBooks = (ListView)array[0];
            ListView lvReaders = (ListView)array[1];
            ListView lvRents = (ListView)array[2];
            int selected = -1;
            if (currentList == "books")
            {
                selected = lvBooks.SelectedIndex;
            }
            else if (currentList == "readers")
            {
                selected = lvReaders.SelectedIndex;
            }
            else if (currentList == "rents")
            {
                selected = lvRents.SelectedIndex;
            }

            Window newWindow = null;
            if (currentList == "books")
            {
                newWindow = new WindowEditBook();
                newWindow = (WindowEditBook)newWindow;
                newWindow.DataContext = new WindowEditBookViewModel(model, books, selected);
            }
            else if (currentList == "readers")
            {
                newWindow = new WindowEditReader();
                newWindow = (WindowEditReader)newWindow;
                newWindow.DataContext = new WindowEditReaderViewModel(model, readers, selected);
            }
            else if (currentList == "rents")
            {
                newWindow = new WindowEditRent();
                newWindow = (WindowEditRent)newWindow;
                newWindow.DataContext = new WindowEditRentViewModel(model, rents, books, readers, selected);
            }
            newWindow.Show();
            newWindow.Closed += UpdateLists;
        }

        public void ButtonDeleteClicked(object sender)
        {
            List<object> array = (List<object>)sender;
            ListView lvBooks = (ListView)array[0];
            ListView lvReaders = (ListView)array[1];
            ListView lvRents = (ListView)array[2];
            if(currentList == "books")
            {
                int selected = lvBooks.SelectedIndex;
                books.RemoveAt(selected);
                model.GetOffice().getBookCollection().RemoveAt(selected);
            }
            else if(currentList == "readers")
            {
                int selected = lvReaders.SelectedIndex;
                readers.RemoveAt(selected);
                model.GetOffice().getReaderCollection().RemoveAt(selected);
            }
            else if(currentList == "rents")
            {
                int selected = lvRents.SelectedIndex;
                rents.RemoveAt(selected);
                model.GetOffice().getRentCollection().RemoveAt(selected);
            }
        }

        public void ButtonLoadClicked(object sender)
        {
            OpenFileDialog ofDialog = new OpenFileDialog();
            ofDialog.Filter = "MARJAN Files (.marjan)|*.marjan|All Files (*.*)|*.*";
            ofDialog.FilterIndex = 1;
            ofDialog.Multiselect = false;

            bool? userClickedOK = ofDialog.ShowDialog();
            if (userClickedOK == true)
            {
                model.DeserializeOffice(ofDialog.FileName);
            }
            UpdateLists();
        }

        public void ButtonSaveClicked(object sender)
        {
            SaveFileDialog ofDialog = new SaveFileDialog();
            ofDialog.Filter = "MARJAN Files (.marjan)|*.marjan";
            ofDialog.FilterIndex = 1;

            bool? userClickedOK = ofDialog.ShowDialog();
            if(userClickedOK == true)
            {
                model.SerializeOffice(ofDialog.FileName);
            }
            UpdateLists();
        }

        public void MainComboBox_SelectionChanged(object sender)
        {
            List<object> array = (List<object>)sender;
            View.MyComboBox myCB = (View.MyComboBox)array[0];
            ListView lvBooks = (ListView)array[1];
            ListView lvReaders = (ListView)array[2];
            ListView lvRents = (ListView)array[3];

            this.currentList = myCB.Text;

            if (this.currentList == "books")
            {
                lvBooks.Visibility = Visibility.Visible;
                lvReaders.Visibility = Visibility.Collapsed;
                lvRents.Visibility = Visibility.Collapsed;
            }
            else if (this.currentList == "readers")
            {
                lvBooks.Visibility = Visibility.Collapsed;
                lvReaders.Visibility = Visibility.Visible;
                lvRents.Visibility = Visibility.Collapsed;
            }
            else if (this.currentList == "rents")
            {
                lvBooks.Visibility = Visibility.Collapsed;
                lvReaders.Visibility = Visibility.Collapsed;
                lvRents.Visibility = Visibility.Visible;
            }

        }

        private void UpdateLists()
        {
            books = new ObservableCollection<Model.Book>(model.GetOffice().getBookCollection());
            readers = new ObservableCollection<Model.Reader>(model.GetOffice().getReaderCollection());
            rents = new ObservableCollection<Model.Rent>(model.GetOffice().getRentCollection());
        }

        private void UpdateLists(object sender, EventArgs e)
        {
            books = new ObservableCollection<Model.Book>(model.GetOffice().getBookCollection());
            readers = new ObservableCollection<Model.Reader>(model.GetOffice().getReaderCollection());
            rents = new ObservableCollection<Model.Rent>(model.GetOffice().getRentCollection());
        }
    }
}
