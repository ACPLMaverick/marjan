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
    public class WindowNewBookViewModel : Common.BindableBase
    {
        private Model.ModelController modelController;
        private ObservableCollection<Model.Book> prop;

        private Model.Book _book;
        public Model.Book book
        {
            get { return _book; }
            set
            {
                if(_book != value)
                {
                    _book = value;
                    OnPropertyChanged("book");
                }
            }
        }

        public Common.DelegateCommand ButtonOKCommand { get; set; }
        public Common.DelegateCommand ButtonCancelCommand { get; set; }

        public WindowNewBookViewModel()
        {
            ButtonOKCommand = new Common.DelegateCommand(ButtonOKClicked);
            ButtonCancelCommand = new Common.DelegateCommand(ButtonCancelClicked);
            this.modelController = null;
            this.prop = null;
        }

        public WindowNewBookViewModel(Model.ModelController controller, ObservableCollection<Model.Book> prop)
        {
            ButtonOKCommand = new Common.DelegateCommand(ButtonOKClicked);
            ButtonCancelCommand = new Common.DelegateCommand(ButtonCancelClicked);
            this.modelController = controller;
            this.prop = prop;
        }

        public void ButtonOKClicked(object sender)
        {
            List<object> sentList = (List<object>)sender;
            TextBox tbTitle = (TextBox)sentList[0];
            TextBox tbAuthor = (TextBox)sentList[1];
            TextBox tbYearReleased = (TextBox)sentList[2];
            string title = tbTitle.Text;
            string author = tbAuthor.Text;
            string yR = tbYearReleased.Text;
            if (title == "") title = "NO TITLE";
            if (author == "") author = "NO AUTHOR";
            if (yR == "") yR = "-1";
            createNewBook(title,author,Convert.ToInt32(yR));
            GetActiveWindow().Close();
        }

        public void ButtonCancelClicked(object sender)
        {
            GetActiveWindow().Close();
        }

        private void createNewBook(string title, string author, int yearReleased)
        {
            book = new Model.Book(title, author, yearReleased, 0);
            //modelController.GetOffice().getBookCollection().Add(book);
            prop.Add(book);
        }
    }
}
