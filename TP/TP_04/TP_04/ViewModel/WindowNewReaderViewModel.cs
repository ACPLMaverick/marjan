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
    public class WindowNewReaderViewModel : Common.BindableBase
    {
        private Model.ModelController modelController;
        private ObservableCollection<Model.Reader> prop;

        private Model.Reader _reader;
        public Model.Reader reader
        {
            get { return _reader; }
            set
            {
                if(_reader != value)
                {
                    _reader = value;
                    OnPropertyChanged("book");
                }
            }
        }

        public Common.DelegateCommand ButtonOKCommand { get; set; }
        public Common.DelegateCommand ButtonCancelCommand { get; set; }

        public WindowNewReaderViewModel()
        {
            ButtonOKCommand = new Common.DelegateCommand(ButtonOKClicked);
            ButtonCancelCommand = new Common.DelegateCommand(ButtonCancelClicked);
            this.modelController = null;
            this.prop = null;
        }

        public WindowNewReaderViewModel(Model.ModelController controller, ObservableCollection<Model.Reader> prop)
        {
            ButtonOKCommand = new Common.DelegateCommand(ButtonOKClicked);
            ButtonCancelCommand = new Common.DelegateCommand(ButtonCancelClicked);
            this.modelController = controller;
            this.prop = prop;
        }

        public void ButtonOKClicked(object sender)
        {
            List<object> sentList = (List<object>)sender;
            TextBox tbName = (TextBox)sentList[0];
            TextBox tbSurname = (TextBox)sentList[1];
            TextBox tbPesel = (TextBox)sentList[2];
            string name = tbName.Text;
            string surname = tbSurname.Text;
            string pesel = tbPesel.Text;
            if (name == "") name = "NO TITLE";
            if (surname == "") surname = "NO AUTHOR";
            if (pesel == "") pesel = "-1";
            createNewBook(name,surname,Convert.ToDouble(pesel));
            GetActiveWindow().Close();
        }

        public void ButtonCancelClicked(object sender)
        {
            GetActiveWindow().Close();
        }

        private void createNewBook(string name, string surname, double pesel)
        {
            reader = new Model.Reader(name, surname, pesel);
            //modelController.GetOffice().getReaderCollection().Add(reader);
            prop.Add(reader);
        }
    }
}
