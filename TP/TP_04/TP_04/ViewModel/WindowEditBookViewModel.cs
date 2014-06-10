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
    public class WindowEditBookViewModel : Common.BindableBase
    {
        private Model.ModelController modelController;
        private ObservableCollection<Model.Book> prop;
        private int selectedItem;

        public string prevTitle {get; set;}
        public string prevAuthor { get; set; }
        public string prevYR{ get; set; }

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

        public WindowEditBookViewModel()
        {
            ButtonOKCommand = new Common.DelegateCommand(ButtonOKClicked);
            ButtonCancelCommand = new Common.DelegateCommand(ButtonCancelClicked);
            this.modelController = null;
            this.prop = null;
            this.selectedItem = -1;
            prevTitle = "none";
            prevAuthor = "none";
            prevYR = "0";
        }

        public WindowEditBookViewModel(Model.ModelController controller, ObservableCollection<Model.Book> prop, int selectedItem)
        {
            ButtonOKCommand = new Common.DelegateCommand(ButtonOKClicked);
            ButtonCancelCommand = new Common.DelegateCommand(ButtonCancelClicked);
            this.modelController = controller;
            this.prop = prop;
            this.selectedItem = selectedItem;
            prevTitle = modelController.GetOffice().getBookCollection().ElementAt(selectedItem).title;
            prevAuthor = modelController.GetOffice().getBookCollection().ElementAt(selectedItem).author;
            prevYR = Convert.ToString(modelController.GetOffice().getBookCollection().ElementAt(selectedItem).yearRelased);
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
            editBook(title,author,Convert.ToInt32(yR));
            System.Windows.Application.Current.Windows[1].Close();
        }

        public void ButtonCancelClicked(object sender)
        {
            System.Windows.Application.Current.Windows[1].Close();
        }

        private void editBook(string title, string author, int yearReleased)
        {
            Model.Book bookCol = modelController.GetOffice().getBookCollection().ElementAt(selectedItem);
            Model.Book bookProp = prop.ElementAt(selectedItem);
            Model.Book myNewBook = new Model.Book(title, author, yearReleased, bookCol.key);
            bookCol.title = title;
            bookCol.author = author;
            bookCol.yearRelased = yearReleased;
            bookProp.title = title;
            bookProp.author = author;
            bookProp.yearRelased = yearReleased;
        }
    }
}
